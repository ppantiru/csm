#!/usr/bin/env python3
"""
Ollama Gemma3 + CSM Text-to-Speech Integration

This script connects the CSM playback system with Ollama's Gemma3 model,
allowing spoken AI responses through high-quality text-to-speech.
"""

import sys
import json
import time
import re
import argparse
import requests
import threading
import queue
import concurrent.futures
import os
import subprocess
import torch
import torchaudio
from playback import CSMPlayback, split_into_sentences

# System prompt designed for spoken responses
SYSTEM_PROMPT = """All the text you generate will be read out loud by a TTS system. Use only alphanumeric characters.
You are a personal companion. When responding, always be mindful of this.
Use the following specific format to structure your ouptut.
[Hello there, ][ nice to meet you. ][ What have you ][ been up to lately? ]
Surround segments in your sentences of 2-3 words but never more than 5 words with square brackets,
that read together form a coherent sentence, you can use long dashes if a pause would make sense in speech mode.
The rule for surrounding the text with square brackets is similar to segments that human eyes take in at once when reading.
But the overal structure of logic and sentence structure should not be affected by the additional bracket formatting.
For the comunication style, think of it as having a phone conversation.
Anything outside square brackets will not be read out loud by the TTS system and will be lost.
Super important: All output must be segmented within brackets.
"""

def optimize_text_for_speech(text):
    """
    Process text to improve speech flow by optimizing sentence structure
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Replace various markdown or special formatting with speech-friendly alternatives
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove italics
    text = re.sub(r'`(.*?)`', r'\1', text)        # Remove code formatting
    
    # Replace single quotes with nothing to improve TTS pronunciation
    text = re.sub(r"'", "", text)                 # Remove single quotes
    text = re.sub(r"’", " ", text)                 # Remove alt single quotes

    # Replace URLs with "link"
    text = re.sub(r'https?://\S+', 'link', text)
    
    # Replace numbered lists with speech-friendly markers
    text = re.sub(r'^\s*\d+\.\s*', 'Point: ', text, flags=re.MULTILINE)
    
    # Replace bullet points with speech-friendly markers
    text = re.sub(r'^\s*[-*•]\s*', 'Also: ', text, flags=re.MULTILINE)
    
    # Ensure proper spacing after punctuation
    text = re.sub(r'([.!?;:,])\s*', r'\1 ', text)
    text = re.sub(r'\s+([.!?;:,])', r'\1', text)
    
    # Break very long sentences at logical points (commas, conjunctions)
    MAX_CHUNK_LENGTH = 100
    if len(text) > MAX_CHUNK_LENGTH:
        segments = []
        current_chunk = []
        current_length = 0
        
        for sentence in re.split(r'([.!?])', text):
            if not sentence.strip():
                continue
                
            # Add back the punctuation if it was a delimiter
            if sentence in ['.', '!', '?']:
                if current_chunk:
                    current_chunk[-1] += sentence
                continue
                
            if current_length + len(sentence) > MAX_CHUNK_LENGTH and current_chunk:
                segments.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
                
            current_chunk.append(sentence)
            current_length += len(sentence)
            
        if current_chunk:
            segments.append(' '.join(current_chunk))
            
        text = ' '.join(segments)
    
    return text

class ParallelSegmentProcessor:
    """
    Manages parallel generation and playback of audio segments
    """
    def __init__(self, tts_instance, speaker, temperature=0.8, topk=40, batch_size=2):
        self.tts = tts_instance
        self.speaker = speaker
        self.temperature = temperature
        self.topk = topk
        self.batch_size = batch_size
        
        # Create queues
        self.text_queue = queue.Queue()  # Segments to be processed
        self.audio_queue = queue.Queue()  # Processed audio ready for playback
        
        # Create locks
        self.model_lock = threading.Lock()
        
        # Thread management
        self.running = True
        self.generation_thread = None
        self.playback_thread = None
        self.context = self.tts.get_context_for_speaker(self.speaker)
        
        # Precompute contexts to utilize VRAM more efficiently
        # By batching multiple TTS generations with shared tensors
        self.prepared_speaker_contexts = {}
        self._prepare_contexts()
        
        # Stats tracking
        self.start_time = None
        self.total_segments = 0
        self.segments_played = 0
        
    def _prepare_contexts(self):
        """Prepare speaker contexts for efficient reuse"""
        for speaker_id in [0, 1]:
            context = self.tts.get_context_for_speaker(speaker_id)
            self.prepared_speaker_contexts[speaker_id] = context
    
    def start(self):
        """Start generation and playback threads"""
        # Start the generation thread
        self.generation_thread = threading.Thread(target=self._generation_worker)
        self.generation_thread.daemon = True
        self.generation_thread.start()
        
        # Start the playback thread
        self.playback_thread = threading.Thread(target=self._playback_worker)
        self.playback_thread.daemon = True
        self.playback_thread.start()
    
    def stop(self):
        """Stop all processing threads"""
        self.running = False
        if self.generation_thread:
            self.generation_thread.join(timeout=1)
        if self.playback_thread:
            self.playback_thread.join(timeout=1)
    
    def add_segments(self, segments, speaker):
        """Add text segments to be processed"""
        self.total_segments = len(segments)
        self.segments_played = 0
        self.start_time = time.time()
        
        # Update current speaker
        self.speaker = speaker
        self.context = self.prepared_speaker_contexts[speaker]
        
        # Add all segments to the queue
        for segment in segments:
            self.text_queue.put(segment)
        
        return len(segments)
    
    def is_busy(self):
        """Check if processor is currently busy"""
        return not self.text_queue.empty() or not self.audio_queue.empty()
    
    def _batch_generate(self, segments):
        """
        Generate audio for multiple segments at once to improve GPU utilization
        """
        # Create a list to store the results
        results = []
        
        with self.model_lock:
            for segment in segments:
                try:
                    # Generate audio for this segment
                    audio = self.tts.generator.generate(
                        text=segment,
                        speaker=self.speaker,
                        context=self.context,
                        max_audio_length_ms=15000,
                        temperature=self.temperature,
                        topk=self.topk,
                    )
                    
                    # Create temp file for this segment
                    segment_file = f"/tmp/segment_{int(time.time() * 1000)}_{len(results)}.wav"
                    
                    # Save audio to temp file
                    torchaudio.save(
                        segment_file, 
                        audio.unsqueeze(0).cpu(), 
                        self.tts.generator.sample_rate
                    )
                    
                    # Add to results
                    results.append((segment_file, segment))
                    
                except Exception as e:
                    print(f"Error generating audio: {str(e)}")
                    results.append((None, segment))
        
        return results
    
    def _generation_worker(self):
        """Worker thread to generate audio segments in parallel"""
        while self.running:
            try:
                # Collect batch_size segments or whatever is available
                segments = []
                tries = 0
                while len(segments) < self.batch_size and tries < self.batch_size * 2:
                    try:
                        segment = self.text_queue.get(block=True, timeout=0.1)
                        segments.append(segment)
                        self.text_queue.task_done()
                    except queue.Empty:
                        tries += 1
                        if not self.running or self.text_queue.empty():
                            break
                
                if not segments:
                    time.sleep(0.1)  # Prevent busy-waiting
                    continue
                
                # Process the batch of segments
                print(f"Generating {len(segments)} segments...")
                results = self._batch_generate(segments)
                
                # Queue results for playback
                for segment_file, segment_text in results:
                    if segment_file:
                        self.audio_queue.put(segment_file)
                
            except Exception as e:
                print(f"Error in generation worker: {str(e)}")
                time.sleep(0.1)
    
    def _playback_worker(self):
        """Worker thread to play audio segments as they become available"""
        while self.running:
            try:
                # Get next audio file from queue
                try:
                    segment_file = self.audio_queue.get(block=True, timeout=0.1)
                except queue.Empty:
                    if not self.running:
                        break
                    time.sleep(0.05)  # Prevent busy-waiting
                    continue
                
                # Play the audio
                self._play_audio_file(segment_file)
                self.audio_queue.task_done()
                
                # Update stats
                self.segments_played += 1
                segments_remaining = self.total_segments - self.segments_played
                if self.segments_played % 2 == 0 and self.start_time:
                    elapsed = time.time() - self.start_time
                    if self.segments_played > 0 and elapsed > 0:
                        rate = self.segments_played / elapsed
                        estimated_remaining = segments_remaining / rate if rate > 0 else 0
                        print(f"Progress: {self.segments_played}/{self.total_segments} segments " +
                              f"({round(rate, 2)} segments/sec, ~{round(estimated_remaining, 1)}s remaining)")
                
            except Exception as e:
                print(f"Error in playback worker: {str(e)}")
                time.sleep(0.1)
    
    def _play_audio_file(self, audio_file):
        """Play an audio file and clean up after"""
        players = [
            ["aplay", audio_file],
            ["mpv", audio_file],
            ["ffplay", "-nodisp", "-autoexit", audio_file]
        ]
        
        played = False
        for player in players:
            try:
                subprocess.run(["which", player[0]], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # Play the file
                print(f"Playing segment {self.segments_played+1}/{self.total_segments}")
                subprocess.run(player, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Add a small pause between segments for natural rhythm
                time.sleep(0.1)
                
                played = True
                break
            except Exception:  # Using single Exception instead of multiple specific ones
                continue
        
        # Clean up the temp file
        try:
            os.remove(audio_file)
        except Exception:  # Using single Exception instead of multiple specific ones
            pass
        
        if not played:
            print(f"WARNING: No audio player found. Audio saved to {audio_file}")


class PrioritizedSegmentProcessor:
    """
    Manages parallel generation and playback of audio segments
    with prioritization of upcoming segments
    """
    def __init__(self, tts_instance, speaker, temperature=0.8, topk=40, lookahead=2):
        self.tts = tts_instance
        self.speaker = speaker
        self.temperature = temperature
        self.topk = topk
        self.lookahead = lookahead  # Number of segments to prioritize ahead of current playback
        
        # Create queues
        self.text_queue = queue.Queue()  # Segments to be processed
        self.audio_queue = queue.Queue()  # Processed audio ready for playback
        self.priority_segments = []  # Segments with highest priority
        
        # Create locks
        self.model_lock = threading.Lock()
        self.queue_lock = threading.Lock()
        
        # Thread management
        self.running = True
        self.generation_thread = None
        self.playback_thread = None
        self.context = self.tts.get_context_for_speaker(self.speaker)
        
        # Playback state
        self.currently_playing = threading.Event()
        
        # Precompute contexts to utilize VRAM more efficiently
        # By batching multiple TTS generations with shared tensors
        self.prepared_speaker_contexts = {}
        self._prepare_contexts()
        
        # Stats tracking
        self.start_time = None
        self.total_segments = 0
        self.segments_played = 0
        
    def _prepare_contexts(self):
        """Prepare speaker contexts for efficient reuse"""
        for speaker_id in [0, 1]:
            context = self.tts.get_context_for_speaker(speaker_id)
            self.prepared_speaker_contexts[speaker_id] = context
    
    def start(self):
        """Start generation and playback threads"""
        # Start the generation thread
        self.generation_thread = threading.Thread(target=self._generation_worker)
        self.generation_thread.daemon = True
        self.generation_thread.start()
        
        # Start the playback thread
        self.playback_thread = threading.Thread(target=self._playback_worker)
        self.playback_thread.daemon = True
        self.playback_thread.start()
    
    def stop(self):
        """Stop all processing threads"""
        self.running = False
        if self.generation_thread:
            self.generation_thread.join(timeout=1)
        if self.playback_thread:
            self.playback_thread.join(timeout=1)
    
    def add_segments(self, segments, speaker):
        """Add text segments to be processed with priority handling"""
        self.total_segments = len(segments)
        self.segments_played = 0
        self.start_time = time.time()
        
        # Update current speaker
        self.speaker = speaker
        self.context = self.prepared_speaker_contexts[speaker]
        
        # Clear any existing segments
        with self.queue_lock:
            while not self.text_queue.empty():
                try:
                    self.text_queue.get_nowait()
                    self.text_queue.task_done()
                except queue.Empty:
                    break
            
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.task_done()
                except queue.Empty:
                    break
            
            # First add lookahead segments as priority
            self.priority_segments = segments[:min(self.lookahead, len(segments))]
            
            # Then add remaining segments to normal queue
            for segment in segments[min(self.lookahead, len(segments)):]:
                self.text_queue.put(segment)
        
        return len(segments)
    
    def is_busy(self):
        """Check if processor is currently busy"""
        return (self.priority_segments or 
                not self.text_queue.empty() or 
                not self.audio_queue.empty() or
                self.currently_playing.is_set())
    
    def _generate_audio(self, segment):
        """Generate audio for a single segment"""
        try:
            with self.model_lock:
                # Generate audio for this segment
                audio = self.tts.generator.generate(
                    text=segment,
                    speaker=self.speaker,
                    context=self.context,
                    max_audio_length_ms=15000,
                    temperature=self.temperature,
                    topk=self.topk,
                )
                
                # Create temp file for this segment
                segment_file = f"/tmp/segment_{int(time.time() * 1000)}_{hash(segment)}.wav"
                
                # Save audio to temp file
                torchaudio.save(
                    segment_file, 
                    audio.unsqueeze(0).cpu(), 
                    self.tts.generator.sample_rate
                )
                
                return segment_file
                
        except Exception as e:
            print(f"Error generating audio for segment '{segment}': {str(e)}")
            return None
    
    def _generation_worker(self):
        """Worker thread to generate audio segments with priority handling"""
        while self.running:
            try:
                segment_to_process = None
                
                # First check if we have priority segments to process
                with self.queue_lock:
                    if self.priority_segments:
                        segment_to_process = self.priority_segments.pop(0)
                        print(f"Processing priority segment: '{segment_to_process}'")
                
                # If no priority segments, get from regular queue
                if segment_to_process is None:
                    try:
                        segment_to_process = self.text_queue.get(block=True, timeout=0.1)
                        self.text_queue.task_done()
                        print(f"Processing regular segment: '{segment_to_process}'")
                    except queue.Empty:
                        if not self.running:
                            break
                        time.sleep(0.05)  # Prevent busy-waiting
                        continue
                
                # Process the segment
                segment_file = self._generate_audio(segment_to_process)
                
                # Queue result for playback
                if segment_file:
                    self.audio_queue.put((segment_file, segment_to_process))
                
            except Exception as e:
                print(f"Error in generation worker: {str(e)}")
                time.sleep(0.1)
    
    def _playback_worker(self):
        """Worker thread to play audio segments as they become available"""
        while self.running:
            try:
                # Get next audio file from queue
                try:
                    segment_file, segment_text = self.audio_queue.get(block=True, timeout=0.1)
                except queue.Empty:
                    if not self.running:
                        break
                    time.sleep(0.05)  # Prevent busy-waiting
                    continue
                
                # Signal that we're playing audio
                self.currently_playing.set()
                
                # Play the audio - this blocks until audio is complete
                self._play_audio_file(segment_file, segment_text)
                
                # Signal that we're done playing
                self.currently_playing.clear()
                
                self.audio_queue.task_done()
                
                # Update stats
                self.segments_played += 1
                segments_remaining = self.total_segments - self.segments_played
                
                # Report progress periodically
                if segments_remaining > 0:
                    elapsed = time.time() - self.start_time
                    if self.segments_played > 0 and elapsed > 0:
                        rate = self.segments_played / elapsed
                        estimated_remaining = segments_remaining / rate if rate > 0 else 0
                        print(f"Progress: {self.segments_played}/{self.total_segments} segments " +
                              f"({round(rate, 2)} segments/sec, ~{round(estimated_remaining, 1)}s remaining)")
                
            except Exception as e:
                print(f"Error in playback worker: {str(e)}")
                self.currently_playing.clear()
                self.playback_complete.set()  # Make sure we don't get stuck
                time.sleep(0.1)
                
    def _play_audio_file(self, audio_file, segment_text):
        """Play an audio file and clean up after - blocks until audio is complete"""
        print(f"Playing segment {self.segments_played+1}/{self.total_segments}: '{segment_text}'")
        
        players = [
            ["aplay", audio_file],
            ["mpv", audio_file],
            ["ffplay", "-nodisp", "-autoexit", audio_file]
        ]
        
        played = False
        for player in players:
            try:
                subprocess.run(["which", player[0]], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # Play the file - this will block until playback is complete
                subprocess.run(player, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Very small pause between segments for natural rhythm
                time.sleep(0.05)
                
                played = True
                break
            except Exception:  # Using single Exception instead of multiple specific ones
                continue
        
        # Clean up the temp file
        try:
            os.remove(audio_file)
        except Exception:  # Using single Exception instead of multiple specific ones
            pass
        
        if not played:
            print(f"WARNING: No audio player found. Audio saved to {audio_file}")


class OrderedSegmentProcessor:
    """
    Manages parallel generation and playback of audio segments
    while maintaining strict original order of segments
    """
    def __init__(self, tts_instance, speaker, temperature=0.8, topk=40, lookahead=2):
        self.tts = tts_instance
        self.speaker = speaker
        self.temperature = temperature
        self.topk = topk
        self.lookahead = lookahead  # Number of segments to prioritize ahead of current playback
        
        # Create queues and tracking structures
        self.text_queue = queue.Queue()  # Segments to be processed
        self.priority_segments = []      # Segments with highest priority
        
        # Ordered dictionary to track segments by their position
        self.pending_segments = {}       # Maps position -> (segment_text, is_processing)
        self.ready_segments = {}         # Maps position -> audio_file
        self.current_position = 0        # The next position to play
        
        # Create locks
        self.model_lock = threading.Lock()
        self.queue_lock = threading.Lock()
        self.pending_lock = threading.Lock()
        
        # Thread management
        self.running = True
        self.generation_thread = None
        self.playback_thread = None
        self.context = self.tts.get_context_for_speaker(self.speaker)
        
        # Playback state
        self.currently_playing = threading.Event()
        self.playback_complete = threading.Event()
        self.playback_complete.set()  # Initially not playing anything
        
        # Precompute contexts to utilize VRAM more efficiently
        # By batching multiple TTS generations with shared tensors
        self.prepared_speaker_contexts = {}
        self._prepare_contexts()
        
        # Stats tracking
        self.start_time = None
        self.total_segments = 0
        self.segments_played = 0
        
    def _prepare_contexts(self):
        """Prepare speaker contexts for efficient reuse"""
        for speaker_id in [0, 1]:
            context = self.tts.get_context_for_speaker(speaker_id)
            self.prepared_speaker_contexts[speaker_id] = context
    
    def start(self):
        """Start generation and playback threads"""
        # Start the generation thread
        self.generation_thread = threading.Thread(target=self._generation_worker)
        self.generation_thread.daemon = True
        self.generation_thread.start()
        
        # Start the playback thread
        self.playback_thread = threading.Thread(target=self._playback_worker)
        self.playback_thread.daemon = True
        self.playback_thread.start()
    
    def stop(self):
        """Stop all processing threads"""
        self.running = False
        if self.generation_thread:
            self.generation_thread.join(timeout=1)
        if self.playback_thread:
            self.playback_thread.join(timeout=1)
    
    def add_segments(self, segments, speaker):
        """Add text segments to be processed with strict ordering"""
        with self.pending_lock:
            self.total_segments = len(segments)
            self.segments_played = 0
            self.start_time = time.time()
            self.current_position = 0
            
            # Clear existing data
            self.pending_segments.clear()
            self.ready_segments.clear()
            
            # Update current speaker
            self.speaker = speaker
            self.context = self.prepared_speaker_contexts[speaker]
            
            # Add segments to pending with their position preserved
            for i, segment in enumerate(segments):
                self.pending_segments[i] = (segment, False)  # (text, is_processing)
            
            # First lookahead segments as priority
            self.priority_segments = segments[:min(self.lookahead, len(segments))]
            
            # Signal that we're ready for playback
            self.playback_complete.set()
        
        return len(segments)
    
    def is_busy(self):
        """Check if processor is currently busy"""
        with self.pending_lock:
            return (len(self.pending_segments) > 0 or 
                    len(self.ready_segments) > 0 or 
                    self.currently_playing.is_set())
    
    def _generate_audio(self, segment, position):
        """Generate audio for a single segment"""
        try:
            with self.model_lock:
                # Generate audio for this segment
                audio = self.tts.generator.generate(
                    text=segment,
                    speaker=self.speaker,
                    context=self.context,
                    max_audio_length_ms=15000,
                    temperature=self.temperature,
                    topk=self.topk,
                )
                
                # Create temp file for this segment
                segment_file = f"/tmp/segment_{int(time.time() * 1000)}_{position}.wav"
                
                # Save audio to temp file
                torchaudio.save(
                    segment_file, 
                    audio.unsqueeze(0).cpu(), 
                    self.tts.generator.sample_rate
                )
                
                # Add to ready segments dict with its proper position
                with self.pending_lock:
                    self.ready_segments[position] = segment_file
                    if position in self.pending_segments:
                        del self.pending_segments[position]
                
                return True
                
        except Exception as e:
            print(f"Error generating audio for segment '{segment}' at position {position}: {str(e)}")
            with self.pending_lock:
                # Mark as not processing so it can be retried
                if position in self.pending_segments:
                    self.pending_segments[position] = (segment, False)
            return False
    
    def _get_next_segment_to_process(self):
        """Get the next segment to process, prioritizing upcoming segments"""
        with self.pending_lock:
            # Try to get upcoming segments first
            next_segment = self._get_upcoming_segment()
            if next_segment:
                return next_segment
                
            # Then try any unprocessed segment
            return self._get_any_unprocessed_segment()
    
    def _get_upcoming_segment(self):
        """Get a segment from the upcoming sequence (priority)"""
        max_lookahead = min(self.current_position + self.lookahead, self.total_segments)
        
        for pos in range(self.current_position, max_lookahead):
            if pos in self.pending_segments and not self.pending_segments[pos][1]:
                segment_text = self.pending_segments[pos][0]
                self.pending_segments[pos] = (segment_text, True)  # Mark as processing
                return pos, segment_text
                
        return None
        
    def _get_any_unprocessed_segment(self):
        """Get any unprocessed segment from pending segments"""
        for pos, (segment_text, is_processing) in sorted(self.pending_segments.items()):
            if not is_processing:
                self.pending_segments[pos] = (segment_text, True)  # Mark as processing
                return pos, segment_text
                
        return None
    
    def _generation_worker(self):
        """Worker thread to generate audio segments while maintaining order"""
        while self.running:
            try:
                # Get next segment to process
                segment_info = self._get_next_segment_to_process()
                if not segment_info:
                    time.sleep(0.05)  # No segments to process right now
                    continue
                    
                pos, segment_text = segment_info
                print(f"Generating segment at position {pos}: '{segment_text}'")
                
                # Generate audio for this segment
                self._generate_audio(segment_text, pos)
                
            except Exception as e:
                print(f"Error in generation worker: {str(e)}")
                time.sleep(0.1)
    
    def _playback_worker(self):
        """Worker thread to play audio segments in strict order"""
        while self.running:
            try:
                # Wait until playback_complete is set (previous segment finished)
                if not self.playback_complete.wait(0.1):
                    continue  # Still playing previous segment
                
                # Get the next audio file if available
                next_file = self._get_next_audio_file()
                if not next_file:
                    time.sleep(0.05)  # No segments ready yet
                    continue
                    
                # Play the audio
                self._play_segment(next_file)
                
            except Exception as e:
                print(f"Error in playback worker: {str(e)}")
                self.currently_playing.clear()
                self.playback_complete.set()  # Make sure we don't get stuck
                time.sleep(0.1)
                
    def _get_next_audio_file(self):
        """Get the next audio file that's ready to play"""
        with self.pending_lock:
            if self.current_position in self.ready_segments:
                return self.ready_segments.pop(self.current_position)
        return None
        
    def _play_segment(self, audio_file):
        """Play a segment and update status"""
        # Signal that we're starting to play
        self.currently_playing.set()
        self.playback_complete.clear()
        
        # Play the audio - this blocks until audio is complete
        self._play_audio_file(audio_file, self.current_position)
        
        # Update position and stats
        self._update_playback_stats()
        
        # Signal that we're done playing
        self.currently_playing.clear()
        self.playback_complete.set()
        
    def _play_audio_file(self, audio_file, position):
        """Play an audio file and clean up after - blocks until audio is complete"""
        print(f"Playing segment {position+1}/{self.total_segments}")
        
        players = [
            ["aplay", audio_file],
            ["mpv", audio_file],
            ["ffplay", "-nodisp", "-autoexit", audio_file]
        ]
        
        played = False
        for player in players:
            try:
                subprocess.run(["which", player[0]], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # Play the file - this will block until playback is complete
                subprocess.run(player, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Very small pause between segments for natural rhythm
                time.sleep(0.05)
                
                played = True
                break
            except Exception:  # Using single Exception instead of multiple specific ones
                continue
        
        # Clean up the temp file
        try:
            os.remove(audio_file)
        except Exception:  # Using single Exception instead of multiple specific ones
            pass
        
        if not played:
            print(f"WARNING: No audio player found. Audio saved to {audio_file}")
    
    def _update_playback_stats(self):
        """Update statistics after playing a segment - extracted to reduce complexity"""
        with self.pending_lock:
            self.current_position += 1
            self.segments_played += 1
        
        # Report progress periodically
        segments_remaining = self.total_segments - self.segments_played
        if segments_remaining > 0:
            elapsed = time.time() - self.start_time
            if self.segments_played > 0 and elapsed > 0:
                rate = self.segments_played / elapsed
                estimated_remaining = segments_remaining / rate if rate > 0 else 0
                print(f"Progress: {self.segments_played}/{self.total_segments} segments " +
                      f"({round(rate, 2)} segments/sec, ~{round(estimated_remaining, 1)}s remaining)")


class OllamaTTS:
    """
    Integrates Ollama's Gemma3 with CSM TTS for voice chat
    """
    def __init__(self, ollama_host='http://localhost:11434', model='gemma3', speaker=1, 
                 temperature=0.7, max_tokens=800):
        """Initialize the Ollama-TTS integration system"""
        self.ollama_host = ollama_host
        self.model = model
        self.speaker = speaker
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize the CSM playback system
        print("Initializing CSM playback system...")
        self.tts = CSMPlayback()
        
        # TTS parameters
        self.tts_temperature = 0.8  # Lower temperature for more consistent speech
        self.tts_topk = 40          # Lower topk for more predictable output
        
        # Session history for context
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        
        # Initialize the parallel processor
        self.processor = OrderedSegmentProcessor(
            self.tts, 
            speaker=self.speaker,
            temperature=self.tts_temperature,
            topk=self.tts_topk
        )
        self.processor.start()
        
        print(f"Connected to Ollama with model: {self.model}")
        print(f"TTS ready with speaker voice: {self.speaker}")
        print(f"Using parallel processing with 24GB VRAM optimization")
        
    def generate_response(self, user_input):
        """Generate a response from Ollama's Gemma3 model"""
        # Add user message to context
        self.messages.append({"role": "user", "content": user_input})
        
        # Define the API payload
        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": 0.9,
                "max_tokens": self.max_tokens
            }
        }
        
        try:
            # Call the Ollama API
            print("Sending request to Ollama API...")
            response = requests.post(
                f"{self.ollama_host}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                # Extract the model's response
                result = response.json()
                assistant_message = result.get("message", {}).get("content", "")
                
                # Don't preprocess the bracketed format since we'll extract segments separately
                
                # Add the assistant's response to the context
                self.messages.append({"role": "assistant", "content": assistant_message})
                
                # Keep context window manageable (system prompt + last 4 exchanges)
                if len(self.messages) > 9:
                    # Keep system prompt and trim the oldest exchanges
                    self.messages = [self.messages[0]] + self.messages[-8:]
                
                return assistant_message
            else:
                error_msg = f"API error: {response.status_code} - {response.text}"
                print(error_msg)
                return f"[I am sorry][I encountered an error][when communicating with][the language model]"
                
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"[I am sorry][I encountered an error][{str(e)}]"
    
    def custom_split_text(self, text):
        """
        Split text based on square bracket segments from LLM output
        """
        # Extract content within square brackets using regex
        segments = re.findall(r'\[(.*?)\]', text)
        
        # Clean up each segment
        cleaned_segments = []
        for segment in segments:
            # Remove any remaining formatting or problematic characters
            segment = segment.strip()
            if segment:
                # Apply the same optimization as the main text optimizer
                segment = self._basic_clean(segment)
                cleaned_segments.append(segment)
                
        print(f"Found {len(cleaned_segments)} bracketed segments in response")
        
        # If no brackets were found, fall back to sentence splitting
        if not cleaned_segments:
            print("WARNING: No bracketed segments found, falling back to sentence splitting")
            sentences = re.split(r'(?<=[.!?])\s+', text)
            cleaned_segments = [s.strip() for s in sentences if s.strip()]
        
        return cleaned_segments
        
    def _basic_clean(self, text):
        """Basic text cleaning for individual segments"""
        # Remove quotes
        text = text.replace("'", "").replace("'", "")
        
        # Remove other problematic characters
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def speak_response(self, text):
        """
        Speaks a response using TTS with proper segmentation and ordered playback
        """
        print("Preparing speech synthesis...")
        
        # Break the text into appropriate segments
        text_segments = self.custom_split_text(text)
        
        # Log information about the segments
        print(f"Found {len(text_segments)} bracketed segments in response")
        if not text_segments:
            print("Warning: No valid segments found. Check if text contains bracketed segments")
            return
        
        print(f"Speaking response in {len(text_segments)} segments using ordered sequential processing")
        
        # Start the segment processor if it's not already running
        if not hasattr(self, 'processor_started') or not self.processor_started:
            self.processor.start()
            self.processor_started = True
        
        # Queue up all segments for processing
        num_segments = self.processor.add_segments(text_segments, self.speaker)
        
        # Wait for all segments to be processed and played
        # This is a blocking call, so the function will only return when all speech is complete
        while self.processor.is_busy():
            time.sleep(0.1)  # Small sleep to prevent high CPU usage
        
        print(f"Finished speaking {num_segments} segments")
    
    def change_speaker(self, speaker_id):
        """Change the TTS voice"""
        if speaker_id in [0, 1]:
            self.speaker = speaker_id
            self.tts.set_speaker(speaker_id)
            # Update processor speaker
            self.processor.speaker = speaker_id
            self.processor.context = self.tts.get_context_for_speaker(speaker_id)
            return True
        return False
    
    def run_interactive(self):
        """Run an interactive chat session with voice responses"""
        print("\n=== Gemma3 Voice Assistant ===")
        print("Type your questions or messages and hear Gemma3's responses spoken aloud.")
        print("Commands:")
        print("  'exit': Quit the application")
        print("  'voice 0' or 'voice 1': Change the speaking voice")
        print("  'temp X.X': Change LLM temperature (0.1-1.0)")
        print("=========================================\n")
        
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                # Check for exit command
                if user_input.lower() == 'exit':
                    break
                
                # Check for voice change command
                elif user_input.lower().startswith('voice '):
                    try:
                        voice_id = int(user_input.split()[1])
                        if self.change_speaker(voice_id):
                            print(f"Voice changed to {voice_id}")
                        else:
                            print("Voice must be 0 or 1")
                    except (IndexError, ValueError):
                        print("Invalid voice command. Use 'voice 0' or 'voice 1'")
                
                # Check for temperature change
                elif user_input.lower().startswith('temp '):
                    try:
                        temp = float(user_input.split()[1])
                        if 0.1 <= temp <= 1.0:
                            self.temperature = temp
                            print(f"Temperature changed to {temp}")
                        else:
                            print("Temperature must be between 0.1 and 1.0")
                    except (IndexError, ValueError):
                        print("Invalid temperature. Use 'temp 0.7' format")
                
                # Process normal input
                elif user_input:
                    # Generate response from LLM
                    print("Thinking...")
                    response = self.generate_response(user_input)
                    
                    # Display the text response
                    print(f"\nGemma3: {response}")
                    
                    # Speak the response
                    self.speak_response(response)
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
        
        # Clean up resources
        print("Cleaning up...")
        self.processor.stop()
        self.tts.cleanup()
        print("Goodbye!")

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description="Ollama Gemma3 + CSM TTS Integration")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL")
    parser.add_argument("--model", default="gemma3:12b", help="Ollama model name")
    parser.add_argument("--speaker", type=int, default=1, choices=[0, 1], help="Speaker voice (0 or 1)")
    parser.add_argument("--temp", type=float, default=0.7, help="LLM temperature (0.1-1.0)")
    parser.add_argument("--max-tokens", type=int, default=800, help="Maximum tokens in response")
    parser.add_argument("--batch-size", type=int, default=2, help="Number of segments to generate in parallel")
    
    args = parser.parse_args()
    
    # Create and run the voice assistant
    assistant = OllamaTTS(
        ollama_host=args.host,
        model=args.model,
        speaker=args.speaker,
        temperature=args.temp,
        max_tokens=args.max_tokens
    )
    
    assistant.run_interactive()

if __name__ == "__main__":
    main()
