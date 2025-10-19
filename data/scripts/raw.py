import argparse
import os
import sys
from pathlib import Path
import librosa
import soundfile as sf


def split_audio_file(input_file, output_dir):
    """
    Split an audio file into 1-second segments.
    
    Args:
        input_file (str): Path to the input .wav file
        output_dir (str): Path to the output directory for segments
    
    Returns:
        int: Number of segments created
    """
    try:
        # Load the audio file
        print(f"Loading audio file: {input_file}")
        audio_data, sample_rate = librosa.load(input_file, sr=None)
        
        # Calculate duration and number of segments
        duration = len(audio_data) / sample_rate
        num_segments = int(duration)  # Truncate to get whole seconds only
        
        print(f"Audio duration: {duration:.2f} seconds")
        print(f"Creating {num_segments} segments (ignoring {duration - num_segments:.2f} seconds)")
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get original filename without extension
        original_name = Path(input_file).stem
        
        # Split into 1-second segments
        samples_per_segment = sample_rate  # 1 second worth of samples
        
        for i in range(num_segments):
            start_sample = i * samples_per_segment
            end_sample = (i + 1) * samples_per_segment
            
            # Extract segment
            segment = audio_data[start_sample:end_sample]
            
            # Create output filename with original name as prefix
            output_filename = f"{original_name}_{i+1:02d}.wav"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save segment
            sf.write(output_path, segment, sample_rate)
            print(f"Created: {output_filename}")
        
        print(f"\nSuccessfully created {num_segments} segments in {output_dir}")
        return num_segments
        
    except Exception as e:
        print(f"Error processing audio file: {e}")
        sys.exit(1)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Split a .wav file into 1-second segments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python raw.py input.wav output_dir
  python raw.py /path/to/audio.wav /path/to/output
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Path to the input .wav file'
    )
    
    parser.add_argument(
        'output_dir',
        help='Path to the output directory for segments'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)
    
    if not args.input_file.lower().endswith('.wav'):
        print(f"Error: Input file must be a .wav file")
        sys.exit(1)
    
    # Run the splitting process
    num_segments = split_audio_file(args.input_file, args.output_dir)
    
    if num_segments > 0:
        print(f"\n✅ Successfully split audio into {num_segments} segments!")
    else:
        print("\n❌ No segments were created")
        sys.exit(1)


if __name__ == "__main__":
    main()
