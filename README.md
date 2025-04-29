# Security Dashboard

A real-time security monitoring system with face recognition capabilities, built using Python and Flask.

## Features

- Real-time face detection and recognition
- Motion detection alerts
- Live camera feed monitoring
- Email notifications for security alerts
- Dashboard for security status monitoring
- Snapshot storage for detected events

## Prerequisites

- Python 3.8 or higher
- Webcam
- Internet connection for email notifications

## Installation

1. Clone the repository:
```bash
git clone https://github.com/deependra06/Security_Dashboard.git
cd Security_Dashboard
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Set up your email credentials in `security/config.py`:
```python
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_USER = 'your-email@gmail.com'
EMAIL_PASSWORD = 'your-app-password'
```

2. Configure camera settings in `security/config.py`:
```python
MOTION_DETECTION_THRESHOLD = 5000
SNAPSHOT_INTERVAL = 5  # seconds
```

## Usage

1. Start the application:
```bash
python web/app.py
```

2. Access the dashboard:
   - Open your browser and go to `http://localhost:5000`
   - Login with your credentials
   - Monitor the security feed and alerts

## Project Structure

```
Security_Dashboard/
├── security/
│   ├── config.py           # Configuration settings
│   ├── email_notifier.py   # Email notification system
│   └── face_detection.py   # Face recognition module
├── web/
│   ├── app.py             # Main application
│   ├── database.py        # Database operations
│   ├── settings_manager.py # Settings management
│   ├── static/            # Static files (CSS, JS)
│   └── templates/         # HTML templates
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Security Features

- Face recognition for authorized personnel
- Motion detection for suspicious activities
- Real-time alerts via email
- Secure user authentication
- Encrypted data storage

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, email [deepaksheoran195@gmail.com] or open an issue in the repository. 
