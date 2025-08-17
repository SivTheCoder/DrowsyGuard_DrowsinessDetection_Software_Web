DrowsyGuard Drowsiness Detection Software
Welcome to DrowsyGuard, a real-time drowsiness detection application built with Streamlit and deployed as a web app. This project uses computer vision techniques, powered by OpenCV and MediaPipe, to monitor eye and facial movements via a webcam to detect signs of drowsiness. It’s designed to enhance safety, particularly for drivers or individuals requiring alertness.
Features

Real-time drowsiness detection using eye aspect ratio (EAR) and facial landmarks.
Webcam integration via Streamlit WebRTC.
User-friendly interface powered by Streamlit.
Cross-platform compatibility with local and cloud deployment.

Prerequisites

Python 3.11 (recommended for compatibility with dependencies).
Git (for cloning the repository).
Internet connection (for installing dependencies and running the app).

Installation

Clone the Repository
git clone https://github.com/yourusername/drowsyguard_drowsinessdetection_software_web.git
cd drowsyguard_drowsinessdetection_software_web


Set Up a Virtual Environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate


Install DependenciesEnsure you have the required packages by installing them from requirements.txt:
pip install -r requirements.txt


The current requirements.txt includes:
streamlit==1.40.1
streamlit-webrtc==0.48.2
opencv-contrib-python==4.11.0.86
numpy>=1.26.0,<2.0.0
mediapipe==0.10.11




Install System Dependencies (if building locally)On Debian-based systems, install additional libraries:
sudo apt-get update
sudo apt-get install -y libglib2.0-0 python3-dev build-essential



Usage

Run the ApplicationLaunch the Streamlit app locally:
streamlit run main.py


Open your browser and navigate to the provided URL (e.g., http://localhost:8501).


Deploy to Streamlit Cloud

Push your code to this GitHub repository.
Visit Streamlit Community Cloud and connect your repository.
The app will deploy automatically. Access it at the generated URL (e.g., drowsyguard-drowsiness-software.streamlit.app).


Features in Action

Allow webcam access when prompted.
The app analyzes your eye movements and alerts you if drowsiness is detected.



Troubleshooting

Dependency Errors: If cv2 or mediapipe is not found, verify the Python version (use 3.11) and ensure requirements.txt is correctly installed. Check the Streamlit Cloud logs for specific errors.
Webcam Issues: Ensure your browser permits webcam access and that streamlit-webrtc is functioning.
Deployment Failures: Common issues include version conflicts (e.g., numpy and opencv-contrib-python). Adjust versions in requirements.txt and reboot the app on Streamlit Cloud.

Project Structure

main.py: The main Streamlit application script.
requirements.txt: Lists Python dependencies.
.python-version: Specifies Python 3.11 for Streamlit Cloud.
packages.txt: Contains system dependencies for apt-get (e.g., libglib2.0-0).

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch: git checkout -b feature-branch.
Make your changes and commit: git commit -m "Description of changes".
Push to the branch: git push origin feature-branch.
Open a Pull Request with a clear description of your changes.

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Built with Streamlit for the web interface.
Utilizes OpenCV and MediaPipe for computer vision.
Inspired by the need for real-time alertness monitoring.

Contact
For questions or support, open an issue in this repository or contact the maintainer at drowsguard.technologies@gmail.com.
