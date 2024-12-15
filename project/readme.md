# Colorythm: Turning Vision into Music

## Overview

Colorythm is a dynamic music control system that integrates computer vision and MIDI signals to create an interactive music experience. The system captures hand movements and colored circles from a video feed, processes these inputs, and sends corresponding MIDI signals to control music software like Ableton Live.

## Features

- **Hand Movement Detection**: Uses MediaPipe to detect hand landmarks and calculate distances between fingers.
- **Circle Detection**: Detects colored circles in the video frame and determines their positions.
- **MIDI Control Mapping**: Provides a utility to initialize and map MIDI controls in Ableton Live.
- **Real-time Processing**: Processes video frames in real-time to detect hand movements and circle positions, sending MIDI signals in real-time based on the detected inputs.

## Project Structure

The project consists of four main files, each playing a crucial role in the overall functionality:

1. **`main.py`**: The entry point of the application. Handles initialization, dependency checks, and orchestrates the main loop.
2. **`camera_detection.py`**: Contains the core logic for detecting hand movements and colored circles from the video feed.
3. **`midi_signals.py`**: Handles the sending of MIDI signals based on detected inputs.
4. **`midi_initializer.py`**: Utility for initializing and mapping MIDI controls in Ableton Live.

## Getting Started

### Prerequisites

- Python 3.11
- OpenCV
- MediaPipe
- Mido
- loopMIDI (for virtual MIDI ports)
- Ableton Live (or any other MIDI-compatible music software)

### Installation

1. **Clone the repository**:

   ```sh
   git clone https://github.com/BenoitSvy/colorythm.git
   cd project

   ```

2. **Install dependencies**:

   pip install opencv-python mediapipe numpy mido

3. **Set up loopMIDI**:

   Download and install loopMIDI.
   Create a virtual MIDI port named "loopMIDI Port".

## Usage

1.  **Initialize MIDI Controls**:

    Run the MIDI initializer to map controls in Ableton Live:

    python midi_initializer.py

    Follow the on-screen instructions to map the MIDI controls.

2.  **Run the Application**:

    Start the main application:

        python main.py

        Ensure your webcam is connected and the video file (e.g., project/vid3.mp4) is available.

3.  **Interact with the System**:

    Use the following commands during runtime:
    a: Send the MIDI sequence based on detected circles.
    q: Quit the program.

## File Descriptions

1. **main.py**

   Purpose: Entry point of the application.
   Functions:
   check_dependencies(): Verifies required libraries are installed.
   check_video_file(video_path): Checks if the webcam is available.
   check_midi_port(virtual_port_name): Verifies the availability of a loopMIDI port.
   main(virtual_port_name, video_path, camera_source): Initializes the system, performs checks, and starts the main loop.

2. **camera_detection.py**

   Purpose: Core logic for detecting hand movements and colored circles.
   Functions:
   detect_circles(frame): Detects circles and their dominant color.
   detect_board(frame): Detects the game board in the video frame.
   get_warped_image(frame, corners): Applies perspective transformation.
   process_hands(frame, output): Processes hand movements and sends MIDI signals.
   main(webcam_source, board_source, virtual_port_name): Main function for detection and processing.

3. **midi_signals.py**

   Purpose: Handles sending MIDI signals based on detected inputs.
   Functions:
   send_hand_controls(output, left_hand_data, right_hand_data, two_hands_data): Sends MIDI control change messages.
   send_midi_matrix(matrix, output, bpm, beg, end): Sends a MIDI matrix to Ableton Live.
   check_midi_port(port_name): Checks the availability of the MIDI port.

4. **midi_initializer.py**

   Purpose: Utility for initializing and mapping MIDI controls.
   Functions:
   check_midi_port(port_name): Checks the availability of the MIDI port.
   initialize_midi_controls(): Initializes MIDI controls with user confirmation.

## Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

    MediaPipe for hand and face detection.
    loopMIDI for virtual MIDI ports.
    Ableton Live for MIDI control and music production.

## Contact

For more information, please contact benoit.savary@polytechnique.edu.
