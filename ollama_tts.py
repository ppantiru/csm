#!/usr/bin/env python3
"""
Ollama Gemma3 + CSM Text-to-Speech Integration

This script connects the CSM playback system with Ollama's Gemma3 model,
allowing spoken AI responses through high-quality text-to-speech.
"""

# Ollama TTS Integration for CSM
# 
# This file provides integration between Ollama LLM services and CSM text-to-speech.
# Key features:
# - Manages communication with Ollama API
# - Optimizes text for natural-sounding speech
# - Segments text based on punctuation for better TTS output
# - Maintains context window for conversational exchanges
#
# Text segmentation strategy:
# 1. Primary segmentation: Split at sentence boundaries (., !, ?)
# 2. Secondary segmentation: Split at commas for more natural pauses
# 3. Tertiary segmentation: Split long segments at conjunctions or word boundaries
# 4. Final cleanup: Normalize spacing and remove empty segments

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
SYSTEM_PROMPT = """All the text you generate will be read out loud by a TTS system.
You are a personal companion. When responding, always be mindful of this.
Use simple, clear language with natural punctuation to help the TTS system.
Avoid overly complex sentences, special characters, or formatting that might be difficult to speak.
For the communication style, think of it as having a phone conversation.
Be concise, warm, and conversational. Aim for short sentences with natural pauses.
Important: Never ever use emojis or the '*' characters as they break the TTS.
"""

def optimize_text_for_speech(text):
    """
    Process text to improve speech flow by optimizing sentence structure
    """
    if not text:
        return ""
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Replace special patterns with spoken equivalents
    replacements = {
        r'(\d+)\.(\d+)': r'\1 point \2',  # 3.14 -> 3 point 14
        r'(\d+)/(\d+)': r'\1 divided by \2',  # 1/2 -> 1 divided by 2
        r'(\d+)\s*\*\s*(\d+)': r'\1 times \2',  # 2*3 -> 2 times 3
        r'(\d+)\s*\+\s*(\d+)': r'\1 plus \2',  # 2+3 -> 2 plus 3
        r'(\d+)\s*-\s*(\d+)': r'\1 minus \2',  # 5-3 -> 5 minus 3
        r'(\d+)\s*=\s*(\d+)': r'\1 equals \2',  # 5=5 -> 5 equals 5
        r'(\d+)%': r'\1 percent',  # 50% -> 50 percent
        r'\$(\d+)': r'\1 dollars',  # $100 -> 100 dollars
        r'\$(\d+)\.(\d+)': r'\1 dollars and \2 cents',  # $10.50 -> 10 dollars and 50 cents
        r'Dr\.': r'Doctor',  # Dr. -> Doctor
        r'Mr\.': r'Mister',  # Mr. -> Mister
        r'Mrs\.': r'Misses',  # Mrs. -> Misses
        r'Ms\.': r'Miss',  # Ms. -> Miss
        r'(\d+):(\d+)': r'\1 \2',  # 3:45 -> 3 45 (for time)
        r'www\.': r'w w w dot',  # www. -> w w w dot
        r'\.com': r' dot com',  # .com -> dot com
        r'\.org': r' dot org',  # .org -> dot org
        r'\.net': r' dot net',  # .net -> dot net
        r'&': r' and ',  # & -> and
        r'\+': r' plus ',  # + -> plus
        r' - ': r', ',  # - -> , (when used as separator)
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
    # Remove excessive whitespace again
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def split_text_for_tts(text):
    """
    Split text into sections for TTS by using punctuation as guidance.
    """
    # First, optimize the text for speech
    text = optimize_text_for_speech(text)
    
    # Split on sentence-ending punctuation (., !, ?)
    sentences = re.split(r'([.!?])\s+', text)
    
    # Re-combine sentence texts with their punctuation
    processed_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            processed_sentences.append(sentences[i] + sentences[i + 1])
        else:
            processed_sentences.append(sentences[i])
    
    # Handle the last element if there's an odd number
    if len(sentences) % 2 != 0:
        processed_sentences.append(sentences[-1])
        
    # Process the sentences into TTS-friendly segments
    segments = _process_sentences_for_tts(processed_sentences)
    
    # Final cleanup: ensure there are no extra spaces and remove empty segments
    clean_segments = []
    for segment in segments:
        if segment.strip():
            # Normalize whitespace - replace any multiple spaces with a single space
            cleaned = re.sub(r'\s+', ' ', segment).strip()
            clean_segments.append(cleaned)
    
    return clean_segments

def _process_sentences_for_tts(sentences):
    """
    Process a list of sentences into TTS-friendly segments
    """
    segments = []
    
    # Break sentences at appropriate points
    for sentence in sentences:
        # Always try to break at commas first for more natural pauses
        comma_parts = re.split(r'(?<=,)\s+', sentence)
        
        # If there are multiple comma parts, use them as separate segments
        if len(comma_parts) > 1:
            segments.extend([p.strip() for p in comma_parts if p.strip()])
            continue
            
        # For sentences without commas but still long, use different strategies
        if len(sentence) <= 100:
            # Short sentences can be used as-is
            segments.append(sentence)
            continue
            
        # Very long sentences: try breaking at conjunctions
        conjunction_parts = _break_at_conjunctions(sentence)
        segments.extend(conjunction_parts)
    
    # Handle any remaining segments that are still too long
    return _break_oversized_segments(segments)

def _break_at_conjunctions(sentence):
    """Break a long sentence at conjunctions"""
    conjunction_parts = re.split(r'\s+(and|but|or|because|so|if|when|while)\s+', sentence)
    
    # Recombine with conjunctions intact
    processed_parts = []
    for i in range(0, len(conjunction_parts), 2):
        if i+1 < len(conjunction_parts):
            processed_parts.append(f"{conjunction_parts[i]} {conjunction_parts[i+1]}")
        else:
            processed_parts.append(conjunction_parts[i])
            
    return [p.strip() for p in processed_parts if p.strip()]

def _split_segment_by_words(segment, max_length=150):
    """Split a segment at word boundaries respecting max length"""
    # First, clean up any extra spaces
    segment = re.sub(r'\s+', ' ', segment).strip()
    
    # If the segment is already short enough, return as is
    if len(segment) <= max_length:
        return [segment]
    
    # Use a more careful word boundary approach
    words = segment.split()
    
    # Handle empty case
    if not words:
        return [segment]
    
    # Initialize result and current segment
    result = []
    current_words = []
    current_length = 0
    
    for word in words:
        word_length = len(word)
        space_length = 1 if current_words else 0
        
        # If this word would push us over the limit, finalize the current segment
        if current_words and (current_length + space_length + word_length > max_length):
            # Add the complete segment to results
            result.append(' '.join(current_words))
            # Start a new segment with the current word
            current_words = [word]
            current_length = word_length
        else:
            # Add the word to the current segment
            current_words.append(word)
            current_length += word_length + space_length
    
    # Don't forget the last segment
    if current_words:
        result.append(' '.join(current_words))
    
    # Verify no empty segments
    return [segment for segment in result if segment.strip()]

def _break_oversized_segments(segments):
    """Break segments that exceed maximum length"""
    final_segments = []
    
    for segment in segments:
        if len(segment) > 150:
            # Break at word boundaries
            final_segments.extend(_split_segment_by_words(segment))
        else:
            final_segments.append(segment)
    
    return final_segments

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
            
            # Set start time for statistics
            self.start_time = time.time()
            
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
        try:
            with self.pending_lock:
                self.current_position += 1
                self.segments_played += 1
        
            # Report progress periodically
            segments_remaining = self.total_segments - self.segments_played
            if segments_remaining > 0 and self.start_time is not None:
                elapsed = time.time() - self.start_time
                if self.segments_played > 0 and elapsed > 0:
                    rate = self.segments_played / elapsed
                    estimated_remaining = segments_remaining / rate if rate > 0 else 0
                    print(f"Progress: {self.segments_played}/{self.total_segments} segments " +
                          f"({round(rate, 2)} segments/sec, ~{round(estimated_remaining, 1)}s remaining)")
        except Exception as e:
            # Catch any errors in statistics calculation to prevent playback issues
            print(f"Warning: Error updating playback stats: {str(e)}")


class OllamaTTS:
    """
    Integrates Ollama's Gemma3 with CSM TTS for voice chat
    """
    def __init__(self, 
                 ollama_host='http://localhost:11434', 
                 model='gemma3', 
                 speaker=1, 
                 temperature=0.7, 
                 max_tokens=800,
                 playback_mock=False):
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
        
        # Mock playback option for testing
        self.playback_mock = playback_mock
        
        # Initialize the parallel processor
        if not self.playback_mock:
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
        try:
            # Add the user's message to the context
            self.messages.append({"role": "user", "content": user_input})
            
            # Call the Ollama API
            response = requests.post(
                f"{self.ollama_host}/api/chat",
                json={
                    "model": self.model,
                    "messages": self.messages,
                    "stream": False,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                response_json = response.json()
                if "message" in response_json and "content" in response_json["message"]:
                    content = response_json["message"]["content"]
                    # Add the assistant's response to the context
                    self.messages.append({"role": "assistant", "content": content})
                    
                    # Keep context window manageable (system prompt + last 4 exchanges)
                    if len(self.messages) > 9:
                        # Keep system prompt and trim the oldest exchanges
                        self.messages = [self.messages[0]] + self.messages[-8:]
                    
                    return content
                else:
                    print("Warning: Unexpected API response structure")
                    return "I encountered an error when communicating with the language model."
            else:
                print(f"API error: {response.status_code}")
                return "I am sorry, I encountered an error when communicating with the language model."
                
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"I am sorry, I encountered an error: {str(e)}"
    
    def custom_split_text(self, text):
        """
        Split text into TTS-friendly segments based on punctuation
        """
        # Use our punctuation-based segmentation function
        segments = split_text_for_tts(text)
        
        print("Split response into {} segments based on punctuation".format(len(segments)))
        
        # Clean each segment
        cleaned_segments = []
        for segment in segments:
            if segment:
                # Apply basic cleaning
                segment = self._basic_clean(segment)
                cleaned_segments.append(segment)
        
        return cleaned_segments
    
    def _basic_clean(self, text):
        """Basic text cleaning for individual segments"""
        # Standardize quotes
        text = text.replace("’", "'")
        text = text.replace("“", '"')
        
        # Remove emojis but keep single quotes
        # text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        
        # Normalize whitespace
        # text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def speak_response(self, text):
        """
        Speaks a response using TTS with proper segmentation and ordered playback
        """
        print("Preparing speech synthesis...")
        
        # Break the text into appropriate segments based on punctuation
        text_segments = self.custom_split_text(text)
        
        # Log information about the segments
        print(f"Speaking response in {len(text_segments)} segments")
        if not text_segments:
            print("Warning: No valid segments found in response")
            return
        
        # Start the segment processor if it's not already running
        if not hasattr(self, 'processor_started') or not self.processor_started:
            if not self.playback_mock:
                self.processor.start()
                self.processor_started = True
        
        # Queue up all segments for processing
        if not self.playback_mock:
            num_segments = self.processor.add_segments(text_segments, self.speaker)
        else:
            num_segments = len(text_segments)
        
        # Wait for all segments to be processed and played
        # This is a blocking call, so the function will only return when all speech is complete
        if not self.playback_mock:
            while self.processor.is_busy():
                time.sleep(0.1)  # Small sleep to prevent high CPU usage
        
        print(f"Finished speaking {num_segments} segments")
    
    def change_speaker(self, speaker_id):
        """Change the TTS voice"""
        if speaker_id in [0, 1]:
            self.speaker = speaker_id
            self.tts.set_speaker(speaker_id)
            # Update processor speaker
            if not self.playback_mock:
                self.processor.speaker = speaker_id
                self.processor.context = self.tts.get_context_for_speaker(speaker_id)
            return True
        return False
    
    def _handle_command(self, input_text):
        """Handle special commands like voice switching"""
        # Not a command
        if not input_text.startswith("/"):
            return False
            
        command = input_text.strip().lower()
        
        # Voice switching
        if command.startswith("/voice"):
            parts = command.split()
            if len(parts) > 1 and parts[1].isdigit():
                speaker_id = int(parts[1])
                if self.change_speaker(speaker_id):
                    print(f"Changed voice to speaker {speaker_id}")
                else:
                    print(f"Failed to change voice to speaker {speaker_id}")
            else:
                print("Usage: /voice [speaker_id]")
            return True
            
        # Help command
        if command == "/help":
            self._show_help()
            return True
            
        # Exit command
        if command in ["/exit", "/quit"]:
            self._handle_exit()
            return True
            
        # Unrecognized command
        print(f"Unknown command: {command}")
        print("Type /help for available commands")
        return True
        
    def _show_help(self):
        """Show help information for available commands"""
        print("\nAvailable commands:")
        print("  /voice [id]  - Change voice to speaker id (1-9)")
        print("  /help        - Show this help message")
        print("  /exit, /quit - Exit the program")
        print("\nJust type normally for conversation.")
    
    def _handle_exit(self):
        """Handle exit command"""
        # Clean up resources
        print("Cleaning up...")
        if not self.playback_mock:
            self.processor.stop()
            self.tts.cleanup()
        print("Goodbye!")
        sys.exit(0)

    def run_interactive(self):
        """Run an interactive chat session with voice responses"""
        print("\nWelcome to Voice Assistant!")
        print("Type your messages, or /help for commands. Press Ctrl+C to exit.")
        
        try:
            while True:
                # Get user input
                try:
                    user_input = input("\nYou: ").strip()
                except EOFError:
                    break
                    
                # Skip empty input
                if not user_input:
                    continue
                
                # Handle special commands
                if self._handle_command(user_input):
                    continue
                
                # Generate and speak response
                print("Assistant: ", end="", flush=True)
                
                response = self.generate_response(user_input)
                print(response)
                
                # Speak the response
                self.speak_response(response)
                
        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt.")
            self._handle_exit()

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description="Ollama Gemma3 + CSM TTS Integration")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL")
    parser.add_argument("--model", default="gemma3", help="Ollama model name")
    parser.add_argument("--speaker", type=int, default=1, help="Speaker voice ID")
    parser.add_argument("--temp", type=float, default=0.7, help="LLM temperature (0.1-1.0)")
    parser.add_argument("--max-tokens", type=int, default=800, help="Maximum tokens in response")
    parser.add_argument("--batch-size", type=int, default=2, help="Number of segments to generate in parallel")
    parser.add_argument("--playback-mock", action="store_true", help="Mock audio playback for testing")
    
    args = parser.parse_args()
    
    assistant = OllamaTTS(
        ollama_host=args.host,
        model=args.model,
        speaker=args.speaker,
        temperature=args.temp,
        max_tokens=args.max_tokens,
        playback_mock=args.playback_mock
    )
    
    assistant.run_interactive()

if __name__ == "__main__":
    main()
