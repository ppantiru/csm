#!/usr/bin/env python3
"""
CSM Playback Script
This script loads the CSM model and provides a text-to-speech interface
that accepts user text inputs and plays them back through audio with low latency.
"""

import os
import sys
import torch
import time
import tempfile
import subprocess
import torchaudio
import re
import threading
import queue
import concurrent.futures
from generator import load_csm_1b, Segment


def load_audio(audio_path, sample_rate):
    """Load and resample audio file"""
    audio_tensor, file_sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=file_sample_rate, new_freq=sample_rate
    )
    return audio_tensor


def split_into_sentences(text):
    """Split text into sentences for faster processing"""
    # Split on sentence boundaries
    sentence_endings = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_endings, text)
    
    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # If a sentence is too long, break it further at commas
    result = []
    for sentence in sentences:
        if len(sentence) > 100:  # Threshold for breaking at commas
            comma_chunks = re.split(r'(?<=,)\s+', sentence)
            result.extend([c.strip() for c in comma_chunks if c.strip()])
        else:
            result.append(sentence)
    
    return result


class AudioChunkProcessor:
    """Process audio chunks sequentially with look-ahead generation"""
    def __init__(self, playback_instance):
        self.playback = playback_instance
        self.text_queue = queue.Queue()  # Queue of text chunks to process
        self.audio_queue = queue.Queue()  # Queue of audio files ready to play
        self.running = True
        self.processing = False
        self.chunk_counter = 0
        
        # Generator lock to prevent concurrent GPU access
        self.generator_lock = threading.Lock()
        
        # Start the generator thread
        self.generator_thread = threading.Thread(target=self._generator_loop)
        self.generator_thread.daemon = True
        self.generator_thread.start()
        
        # Start the player thread
        self.player_thread = threading.Thread(target=self._player_loop)
        self.player_thread.daemon = True
        self.player_thread.start()
    
    def _generator_loop(self):
        """Generate audio chunks one at a time but ahead of playback"""
        while self.running:
            try:
                if not self.text_queue.empty():
                    self.processing = True
                    
                    # Get the next chunk from the queue
                    text, speaker, chunk_id = self.text_queue.get(timeout=0.2)
                    
                    # Generate audio for this chunk
                    audio_file = self.playback._generate_audio_chunk(text, speaker, chunk_id)
                    
                    # If successful, add to the playback queue
                    if audio_file:
                        self.audio_queue.put((audio_file, chunk_id))
                    
                    self.text_queue.task_done()
                    
                    # Don't mark as not processing as there might be more chunks in the queue
                    if self.text_queue.empty():
                        self.processing = False
                else:
                    # Sleep briefly if there's nothing to process
                    time.sleep(0.1)
            except queue.Empty:
                # This is expected, just continue
                pass
            except Exception as e:
                print(f"Error in generator loop: {e}")
                self.processing = False
    
    def _player_loop(self):
        """Play audio files in the correct order"""
        next_chunk_to_play = 0
        pending_chunks = {}
        
        while self.running:
            try:
                # Get a chunk from the queue if available
                try:
                    audio_file, chunk_id = self.audio_queue.get(timeout=0.2)
                    pending_chunks[chunk_id] = audio_file
                except queue.Empty:
                    # No new chunks, continue checking pending chunks
                    pass
                
                # Play chunks in order
                while next_chunk_to_play in pending_chunks:
                    audio_file = pending_chunks.pop(next_chunk_to_play)
                    self._play_audio_file(audio_file)
                    next_chunk_to_play += 1
                    
                    # If this was the last chunk and no more text to process
                    if next_chunk_to_play > self.chunk_counter - 1 and self.text_queue.empty():
                        self.processing = False
                
                # If there's nothing to play, sleep briefly
                if not pending_chunks and self.audio_queue.empty():
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Error in player loop: {e}")
    
    def _play_audio_file(self, audio_file):
        """Play a single audio file and clean up after"""
        players = [
            ["aplay", audio_file],
            ["mpv", audio_file],
            ["ffplay", "-nodisp", "-autoexit", audio_file]
        ]
        
        for player in players:
            try:
                subprocess.run(["which", player[0]], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # Play and wait for completion
                subprocess.run(player, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Try to remove the temporary file after playing
                try:
                    os.remove(audio_file)
                except (FileNotFoundError, OSError):
                    pass
                
                return
            except subprocess.CalledProcessError:
                continue
        
        print(f"WARNING: No audio player found. Audio saved to {audio_file}")
    
    def add_text(self, text, speaker):
        """Add text to be processed, returns chunk_id"""
        chunk_id = self.chunk_counter
        self.text_queue.put((text, speaker, chunk_id))
        self.chunk_counter += 1
        self.processing = True
        return chunk_id
    
    def add_chunk_batch(self, chunks, speaker):
        """Add a batch of text chunks with the same speaker"""
        first_chunk_id = self.chunk_counter
        for chunk in chunks:
            self.text_queue.put((chunk, speaker, self.chunk_counter))
            self.chunk_counter += 1
        self.processing = True
        return first_chunk_id, self.chunk_counter - 1
    
    def is_busy(self):
        """Check if the processor is currently busy"""
        return (not self.text_queue.empty() or 
                not self.audio_queue.empty() or
                self.processing)
    
    def stop(self):
        """Stop the audio processor"""
        self.running = False


class CSMPlayback:
    """
    Main class for loading the CSM model and handling text-to-speech generation
    """
    def __init__(self):
        # Detect available device
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        print(f"Using device: {self.device}")
        
        # Load the model
        print("Loading CSM model...")
        self.generator = load_csm_1b(device=self.device)
        print("Model loaded successfully!")
        
        # Create a temporary directory for audio output
        self.temp_dir = tempfile.mkdtemp()
        
        # Load pre-defined reference voices from sample files
        self.load_reference_segments()
        
        # Initialize current speaker
        self.current_speaker = 0
        
        # Model lock to prevent concurrent GPU access
        self.model_lock = threading.Lock()
        
        # Start audio chunk processor
        print("Initializing audio processor...")
        self.audio_processor = AudioChunkProcessor(self)
        
    def load_reference_segments(self):
        """Load the reference voice segments from sample audio files"""
        print("Loading reference voice samples...")
        
        # Define the sample data
        speakers = [0, 1, 0, 0, 0, 0]
        transcripts = [
            "I can't get the neurotoxin into your head any faster.",
            "I'll use lasers to inscribe a line down the center of the facility, and one half will be where you live, and I'll live in the other half.",
            "It also says you are adopted, so that's funny too.",
            "It says so right here in your personnel file: unlikable, liked by no one, a bitter unlikable owner whose passing shall not be mourned.",
            "The rocket really is the way to go.",
            "Well, you've managed to destroy that part of me."
        ]
        audio_paths = [
            "a1.wav",
            "a2.wav",
            "a3.wav",
            "a4.wav",
            "a5.wav",
            "a6.wav"
        ]
        
        # Load audio segments
        self.reference_segments = [
            Segment(
                text=transcript,
                speaker=speaker,
                audio=load_audio(audio_path, self.generator.sample_rate)
            )
            for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
        ]
        
        # Group segments by speaker
        self.voice_0_segments = [seg for seg in self.reference_segments if seg.speaker == 0]
        self.voice_1_segments = [seg for seg in self.reference_segments if seg.speaker == 1]
        
        print(f"Loaded {len(self.reference_segments)} voice samples")
        print(f"Voice 0: {len(self.voice_0_segments)} samples, Voice 1: {len(self.voice_1_segments)} samples")
    
    def get_context_for_speaker(self, speaker):
        """Get appropriate context segments for the specified speaker"""
        if speaker == 0:
            # Use all voice 0 samples as context
            return self.voice_0_segments
        else:
            # Use voice 1 samples as context
            return self.voice_1_segments
    
    def generate_and_play(self, text, speaker=None):
        """
        Break text into sentences and process them with look-ahead
        """
        if speaker is None:
            speaker = self.current_speaker
            
        # Split the text into sentences
        sentences = split_into_sentences(text)
        
        if not sentences:
            print("No valid text to generate speech from.")
            return
            
        print(f"Processing '{text}' as {len(sentences)} chunks (Speaker {speaker})")
        
        # Queue all sentences for sequential processing with look-ahead
        self.audio_processor.add_chunk_batch(sentences, speaker)
        
        return True
    
    def _generate_audio_chunk(self, text, speaker, chunk_id):
        """Generate audio for a single text chunk and return the file path"""
        print(f"Generating chunk {chunk_id}: '{text}' (Speaker {speaker})")
        
        # Create a unique temp file for this chunk
        temp_file = os.path.join(self.temp_dir, f"chunk_{chunk_id}_{int(time.time() * 1000)}.wav")
        
        # Get appropriate context segments for the current speaker
        context = self.get_context_for_speaker(speaker)
        
        # Generate audio using the CSM model with the voice samples as context
        # Use a lock to prevent concurrent access to the GPU
        try:
            with self.model_lock:
                audio = self.generator.generate(
                    text=text,
                    speaker=speaker,
                    context=context,
                    max_audio_length_ms=10_000,
                    temperature=0.9,
                    topk=50,
                )
            
            # Save audio to temporary file
            torchaudio.save(temp_file, audio.unsqueeze(0).cpu(), self.generator.sample_rate)
            print(f"Generated chunk {chunk_id}")
            
            return temp_file
                
        except Exception as e:
            print(f"Error generating audio chunk {chunk_id}: {str(e)}")
            return None
    
    def set_speaker(self, speaker):
        """Set the current speaker (0 or 1)"""
        if speaker in [0, 1]:
            self.current_speaker = speaker
            return True
        return False
    
    def is_processing(self):
        """Check if the system is currently processing audio"""
        return self.audio_processor.is_busy()
    
    def cleanup(self):
        """Clean up resources"""
        # Stop the audio processor
        self.audio_processor.stop()
        
        # Clean up temp directory
        for file in os.listdir(self.temp_dir):
            try:
                os.remove(os.path.join(self.temp_dir, file))
            except OSError:
                pass
                
        try:
            os.rmdir(self.temp_dir)
        except OSError:
            pass


def main():
    """Main function to run the playback interface"""
    print("CSM Playback - Low Latency Text-to-Speech")
    print("----------------------------------------")
    
    # Initialize the CSM playback system
    csm = CSMPlayback()
    
    try:
        print("\nReady for input. Type 'exit' to quit, or 'speaker X' to change speaker (0 or 1).")
        
        while True:
            # Get user input
            user_input = input(f"\n[Speaker {csm.current_speaker}] > ")
            
            # Check for exit command
            if user_input.lower() == 'exit':
                break
                
            # Check for speaker change command
            elif user_input.lower().startswith('speaker '):
                try:
                    new_speaker = int(user_input.split()[1])
                    if csm.set_speaker(new_speaker):
                        print(f"Speaker changed to {new_speaker}")
                    else:
                        print("Speaker must be 0 or 1")
                except (IndexError, ValueError):
                    print("Invalid speaker command. Use 'speaker 0' or 'speaker 1'")
            
            # Generate and play audio for the input text
            elif user_input.strip():
                csm.generate_and_play(user_input)
                
                # Provide feedback to user that processing is happening
                last_status = ""
                status_counter = 0
                
                while csm.is_processing():
                    status_chars = ["-", "\\", "|", "/"]
                    status = f"Processing {status_chars[status_counter % 4]} "
                    if status != last_status:
                        print(status, end="\r", flush=True)
                        last_status = status
                    status_counter += 1
                    time.sleep(0.1)
                
                print(" " * 30, end="\r")  # Clear the status line
                print("Playback complete.")
    
    except KeyboardInterrupt:
        print("\nExiting...")
    
    finally:
        # Clean up resources
        csm.cleanup()
        print("Goodbye!")


if __name__ == "__main__":
    main()
