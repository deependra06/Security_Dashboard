import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
import time
import os
import sys

# Add the parent directory to the path so we can import the database module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from web.database import EmailLog, get_db

class EmailNotifier:
    def __init__(self, config):
        self.config = config
        self.last_email_time = {}  # Track last email sent time for each type
        self.email_interval = 60  # Email interval in seconds (1 minute)
        self.pending_emails = {}  # Track pending email notifications
        self.emails_enabled = False  # Flag to control email sending
        self.verify_config()

    def log_email(self, detection_type, detection_time, confidence, snapshot_name, status):
        """Log email information to database"""
        try:
            db = next(get_db())
            email_log = EmailLog(
                date=detection_time.strftime('%Y-%m-%d'),
                time=detection_time.strftime('%H:%M:%S'),
                recipient=self.config['receiver_email'],
                subject=f"{detection_type} Detection",
                status=status
            )
            db.add(email_log)
            db.commit()
        except Exception as e:
            print(f"Error logging email information: {str(e)}")

    def enable_emails(self):
        """Enable email notifications"""
        self.emails_enabled = True
        print("Email notifications enabled")
    
    def disable_emails(self):
        """Disable email notifications"""
        self.emails_enabled = False
        print("Email notifications disabled")
    
    def is_emails_enabled(self):
        """Check if emails are currently enabled"""
        return self.emails_enabled

    def send_notification(self, detection_type, confidence, snapshot_path=None):
        """Send email notification with retry mechanism"""
        if not self.emails_enabled:
            print(f"Email notifications are currently disabled. Skipping {detection_type} notification.")
            return
            
        current_time = datetime.now()
        snapshot_name = os.path.basename(snapshot_path) if snapshot_path else "N/A"
        
        # Check if enough time has passed since last email for this detection type
        if detection_type in self.last_email_time:
            time_since_last_email = (current_time - self.last_email_time[detection_type]).total_seconds()
            if time_since_last_email < self.email_interval:
                # Store the pending email information
                self.pending_emails[detection_type] = {
                    'confidence': confidence,
                    'snapshot_path': snapshot_path,
                    'detection_time': current_time
                }
                print(f"Email queued for {detection_type} - Will send in {self.email_interval - time_since_last_email:.1f} seconds")
                self.log_email(detection_type, current_time, confidence, snapshot_name, "Queued")
                return
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config['sender_email']
            msg['To'] = self.config['receiver_email']
            
            if detection_type == 'motion':
                subject = f"Motion Detected - Confidence: {confidence:.2f}"
                body = f"Motion was detected at {current_time.strftime('%Y-%m-%d %H:%M:%S')} with confidence {confidence:.2f}"
            else:  # unknown face
                subject = f"Unknown Face Detected - Confidence: {confidence:.2f}"
                body = f"An unknown face was detected at {current_time.strftime('%Y-%m-%d %H:%M:%S')} with confidence {confidence:.2f}"
            
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach snapshot if available
            if snapshot_path and os.path.exists(snapshot_path):
                with open(snapshot_path, 'rb') as f:
                    img = MIMEImage(f.read())
                    img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(snapshot_path))
                    msg.attach(img)
            
            # Send email with retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                        server.starttls()
                        server.login(self.config['sender_email'], self.config['sender_password'])
                        server.send_message(msg)
                    print(f"Email notification sent for {detection_type} at {current_time.strftime('%H:%M:%S')}")
                    self.last_email_time[detection_type] = current_time
                    self.log_email(detection_type, current_time, confidence, snapshot_name, "Sent")
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"Failed to send email after {max_retries} attempts: {str(e)}")
                        self.log_email(detection_type, current_time, confidence, snapshot_name, "Failed")
                    else:
                        print(f"Retrying email send (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(2)
            
        except Exception as e:
            print(f"Error preparing email: {str(e)}")
            self.log_email(detection_type, current_time, confidence, snapshot_name, "Failed")

    def verify_config(self):
        """Verify email configuration and test connection"""
        try:
            with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                server.starttls()
                server.login(self.config['sender_email'], self.config['sender_password'])
                print("Email configuration verified successfully")
        except smtplib.SMTPAuthenticationError:
            print("\nERROR: Email authentication failed. Please check:")
            print("1. Make sure 2-factor authentication is enabled in your Google Account")
            print("2. Generate an App Password from Google Account settings")
            print("3. Use the generated App Password in the code")
            print("4. Make sure you're using the correct email address")
            raise
        except Exception as e:
            print(f"\nERROR: Failed to verify email configuration: {str(e)}")
            raise 