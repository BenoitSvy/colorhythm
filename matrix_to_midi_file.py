import mido
from mido import MidiFile, MidiTrack, Message
import os
from create_matrix import create_matrix  # Import the matrix from the creating matrix file

def matrix_to_midi_file(k,bpm):

    matrix=create_matrix(k,bpm)

    # Create a new MIDI file
    mid = MidiFile(ticks_per_beat=500)

    # Add a new track
    track = MidiTrack()
    mid.tracks.append(track)

    # Sort the matrix by absolute start time to ensure notes are added in the correct order
    matrix.sort(key=lambda x: x[2])

    # Variable to keep track of the current time in the MIDI track
    current_time = 0

    # Create a list to store all events (note_on and note_off)
    all_events = []

    # Iterate over the matrix and create note_on and note_off events
    for note in matrix:
        pitch, velocity, start_time, duration = note
        all_events.append((start_time, 'note_on', pitch, velocity))
        all_events.append((start_time + duration, 'note_off', pitch, velocity))

    # Sort all events by their time
    all_events.sort(key=lambda x: x[0])

    # Add events to the track
    for event_time, event_type, pitch, velocity in all_events:
        # Calculate the relative time
        relative_time = event_time - current_time
        
        # Add the event to the track
        track.append(Message(event_type, note=pitch, velocity=velocity, time=relative_time))
        
        # Update the current time
        current_time = event_time

    # Save the MIDI file in the same directory as this script
    script_dir = os.path.dirname(__file__)
    output_file_path = os.path.join(script_dir, 'output.mid')
    mid.save(output_file_path)



if __name__ == "__main__":
    matrix_to_midi_file(k=2,bpm=120)
