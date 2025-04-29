import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import sys

# Add the parent directory to the path so we can import the database module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from web.database import KnownFace, UnknownFace, MotionEvent, get_db
from security.email_notifier import EmailNotifier
from security.config import (
    UPLOADS_DIR, SNAPSHOTS_MOTION_DIR, SNAPSHOTS_UNKNOWN_DIR
)

class detection():
    def __init__(self):
        self.nameList = []
        self.lastEntryTime = {}  # Dictionary to store last entry time for each person
        self.prev_frame = None  # For motion detection
        self.motion_history = []  # Store recent motion frames
        self.motion_history_size = 5  # Number of frames to consider for motion history
        
        # Email configuration
        email_config = {
            'sender_email': 'deepaksheoran195@gmail.com',
            'sender_password': '',
            'receiver_email': 'dkumar05003@gmail.com',
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587
        }
        
        # Initialize email notifier
        self.email_notifier = EmailNotifier(email_config)
        
        images = []
        self.classNames = []
        myList = os.listdir(UPLOADS_DIR)
        for cl in myList:
            try:
                curImg = cv2.imread(os.path.join(UPLOADS_DIR, cl))
                if curImg is not None:
                    images.append(curImg)
                    self.classNames.append(os.path.splitext(cl)[0])
                else:
                    print(f"Warning: Could not read image {cl}")
            except Exception as e:
                print(f"Error processing image {cl}: {str(e)}")
        
        if not images:
            print("Warning: No valid images found in the directory")
        
        self.encodeListKnown = self.findEncodings(images)

    def findEncodings(self,images):
        encodeList = []
        for img in images:
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encode = face_recognition.face_encodings(img)[0]
                encodeList.append(encode)
            except Exception as e:
                print(f"Error encoding image: {str(e)}")
        return encodeList
    
    def markAttendance(self,name, confidence=None):
        current_time = datetime.now()
        
        # Check if this is the first entry for this person or if 2 seconds have passed
        if name not in self.lastEntryTime or (current_time - self.lastEntryTime[name]).total_seconds() >= 2:
            # Save snapshot with face location
            snapshot_name = "N/A"
            
            # Log to database
            db = next(get_db())
            known_face = KnownFace(
                name=name,
                date=current_time.strftime('%Y-%m-%d'),
                time=current_time.strftime('%H:%M:%S'),
                confidence=confidence if confidence is not None else 0.0,
                snapshot=snapshot_name,
                status='Detected'
            )
            db.add(known_face)
            db.commit()
            
            print(f"Entry marked for {name} at {current_time.strftime('%H:%M:%S')} (Confidence: {confidence:.2f if confidence is not None else 'N/A'})")
            self.lastEntryTime[name] = current_time

    def save_snapshot(self, img, detection_type, confidence, box=None):
        current_time = datetime.now()
        timestamp = current_time.strftime('%Y%m%d_%H%M%S')
        
        # Create a copy of the image to add annotations
        snapshot = img.copy()
        
        # Add detection box if provided
        if box:
            x1, y1, x2, y2 = box
            color = (0, 0, 255)  # Red for both motion and unknown faces
            cv2.rectangle(snapshot, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(snapshot, (x1, y2-35), (x2, y2), color, cv2.FILLED)
        
        # Add type and confidence text
        if detection_type == 'motion':
            folder = SNAPSHOTS_MOTION_DIR
            filename = f'motion_{timestamp}_{confidence:.2f}.jpg'
            text = f"Motion Detected ({confidence:.2f})"
        else:  # unknown face
            folder = SNAPSHOTS_UNKNOWN_DIR
            filename = f'unknown_{timestamp}_{confidence:.2f}.jpg'
            text = f"Unknown Face ({confidence:.2f})"
        
        # Add text at the top of the image
        cv2.putText(snapshot, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Add timestamp
        timestamp_text = current_time.strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(snapshot, timestamp_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        filepath = os.path.join(folder, filename)
        cv2.imwrite(filepath, snapshot)
        print(f"Saved {detection_type} snapshot: {filepath}")
        return filepath  # Return the filepath for email attachment

    def markUnknownFace(self, confidence, img, faceLoc=None):
        current_time = datetime.now()
        name = 'Unknown'
        
        # Check if 2 seconds have passed since last entry
        if name not in self.lastEntryTime or (current_time - self.lastEntryTime[name]).total_seconds() >= 2:
            # Save snapshot with face location
            snapshot_name = "N/A"
            if faceLoc:
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                snapshot_path = self.save_snapshot(img, 'unknown_face', confidence, (x1, y1, x2, y2))
                snapshot_name = os.path.basename(snapshot_path)
                # Send email notification
                self.email_notifier.send_notification('unknown_face', confidence, snapshot_path)
            
            # Log to database
            db = next(get_db())
            unknown_face = UnknownFace(
                date=current_time.strftime('%Y-%m-%d'),
                time=current_time.strftime('%H:%M:%S'),
                confidence=confidence,
                snapshot=snapshot_name,
                status='Detected'
            )
            db.add(unknown_face)
            db.commit()
            
            print(f"Unknown face detected at {current_time.strftime('%H:%M:%S')} (Confidence: {confidence:.2f})")
            self.lastEntryTime[name] = current_time

    def markMotion(self, confidence, img, motion_box=None):
        current_time = datetime.now()
        name = 'Motion Event'
        
        # Check if 2 seconds have passed since last entry
        if name not in self.lastEntryTime or (current_time - self.lastEntryTime[name]).total_seconds() >= 2:
            # Save snapshot with motion box
            snapshot_name = "N/A"
            if motion_box:
                snapshot_path = self.save_snapshot(img, 'motion', confidence, motion_box)
                snapshot_name = os.path.basename(snapshot_path)
                # Send email notification
                self.email_notifier.send_notification('motion', confidence, snapshot_path)
            
            # Log to database
            db = next(get_db())
            motion_event = MotionEvent(
                date=current_time.strftime('%Y-%m-%d'),
                time=current_time.strftime('%H:%M:%S'),
                confidence=confidence,
                snapshot=snapshot_name,
                status='Detected'
            )
            db.add(motion_event)
            db.commit()
            
            print(f"Motion detected at {current_time.strftime('%H:%M:%S')} (Confidence: {confidence:.2f})")
            self.lastEntryTime[name] = current_time

    def calculate_motion_confidence(self, contours, frame_delta):
        if not contours:
            return 0.0
        
        # Calculate base confidence from contour areas
        total_area = sum(cv2.contourArea(contour) for contour in contours)
        area_confidence = min(1.0, total_area / 10000.0)  # Normalize to 0-1 range
        
        # Calculate motion intensity from frame difference
        motion_intensity = np.mean(frame_delta) / 255.0  # Normalize to 0-1 range
        
        # Calculate number of moving objects
        num_objects = len(contours)
        object_confidence = min(1.0, num_objects / 5.0)  # Normalize to 0-1 range
        
        # Calculate motion spread (how distributed the motion is)
        if len(contours) > 1:
            centers = []
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centers.append((cx, cy))
            
            if len(centers) > 1:
                # Calculate average distance between centers
                total_distance = 0
                count = 0
                for i in range(len(centers)):
                    for j in range(i+1, len(centers)):
                        total_distance += np.sqrt((centers[i][0] - centers[j][0])**2 + 
                                                (centers[i][1] - centers[j][1])**2)
                        count += 1
                if count > 0:
                    avg_distance = total_distance / count
                    spread_confidence = min(1.0, avg_distance / 300.0)  # Normalize to 0-1 range
                else:
                    spread_confidence = 0.0
            else:
                spread_confidence = 0.0
        else:
            spread_confidence = 0.0
        
        # Combine all factors with weights
        confidence = (
            0.4 * area_confidence +      # Size of motion
            0.3 * motion_intensity +     # Intensity of motion
            0.2 * object_confidence +    # Number of moving objects
            0.1 * spread_confidence      # Spread of motion
        )
        
        return min(1.0, max(0.0, confidence))  # Ensure confidence is between 0 and 1

    def process_frame(self, img):
        """Process a single frame and return the processed image"""
        try:
            # Add email status to the display first
            email_status = "Emails: ON" if self.email_notifier.is_emails_enabled() else "Emails: OFF"
            cv2.putText(img, email_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Convert to grayscale for motion detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # Initialize previous frame if needed
            if self.prev_frame is None:
                self.prev_frame = gray
                return img
            
            # Calculate absolute difference between frames
            frame_delta = cv2.absdiff(self.prev_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check for motion and calculate confidence
            motion_detected = False
            motion_box = None
            motion_confidence = 0.0
            
            if contours:
                motion_detected = True
                # Get the bounding box of the largest motion
                largest_contour = max(contours, key=cv2.contourArea)
                (x, y, w, h) = cv2.boundingRect(largest_contour)
                motion_box = (x, y, x + w, y + h)
                # Calculate motion confidence
                motion_confidence = self.calculate_motion_confidence(contours, frame_delta)
            
            # Update previous frame
            self.prev_frame = gray
                
            # Resize image for faster face detection
            imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            
            # Detect faces in the current frame
            facesCurFrame = face_recognition.face_locations(imgS, model="hog")  # Use HOG model for faster detection
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
            
            # If motion is detected but no faces are found
            if motion_detected and not facesCurFrame:
                self.markMotion(motion_confidence, img, motion_box)  # Pass motion box for snapshot
                # Draw red box around motion area
                if motion_box:
                    x1, y1, x2, y2 = motion_box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, f"Motion Detected ({motion_confidence:.2f})", (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Process detected faces
            if facesCurFrame:
                # Check if we have any known faces to compare against
                if not self.encodeListKnown:
                    # No known faces, mark all detected faces as unknown
                    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                        # Use a default confidence value since we can't compare
                        confidence = 0.5
                        name = 'Unknown'
                        # Draw face box and label
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 0, 255), cv2.FILLED)
                        cv2.putText(img, f"{name} ({confidence:.2f})", (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                        # Mark unknown face in database (non-blocking)
                        self.markUnknownFace(confidence, img, faceLoc)
                else:
                    # We have known faces, proceed with normal face recognition
                    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                        # Compare face with known faces
                        matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace, tolerance=0.6)  # Increased tolerance for better matching
                        faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                        matchIndex = np.argmin(faceDis)
                        confidence = 1 - faceDis[matchIndex]  # Convert distance to confidence score
                        
                        # Draw face box and label first
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                        
                        if matches[matchIndex] and faceDis[matchIndex] < 0.50:
                            name = self.classNames[matchIndex].upper()
                            color = (0, 255, 0)  # Green for known faces
                            # Draw green box for known face
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                            cv2.rectangle(img, (x1, y2-35), (x2, y2), color, cv2.FILLED)
                            cv2.putText(img, f"{name} ({confidence:.2f})", (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                            # Mark attendance in database (non-blocking)
                            self.markAttendance(name, confidence)
                        else: 
                            name = 'Unknown'
                            color = (0, 0, 255)  # Red for unknown faces
                            # Draw red box for unknown face
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                            cv2.rectangle(img, (x1, y2-35), (x2, y2), color, cv2.FILLED)
                            cv2.putText(img, f"{name} ({confidence:.2f})", (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                            # Mark unknown face in database (non-blocking)
                            self.markUnknownFace(confidence, img, faceLoc)
            
            return img
            
        except Exception as e:
            print(f"Error in process_frame: {str(e)}")
            # Add email status even if there's an error
            email_status = "Emails: ON" if self.email_notifier.is_emails_enabled() else "Emails: OFF"
            cv2.putText(img, email_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return img  # Return original image if there's an error

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        print("Starting face detection...")
        print("Press 'q' to quit")
        print("Press 'e' to toggle email notifications")
        print("Press 's' to show email status")
 
        while(cap.isOpened()):
            success, img = cap.read()
            if not success:
                print("Error: Could not read from webcam")
                break
                
            # Process the frame
            img = self.process_frame(img)
            
            cv2.imshow('Webcam',img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('e'):
                if self.email_notifier.is_emails_enabled():
                    self.email_notifier.disable_emails()
                else:
                    self.email_notifier.enable_emails()
            elif key == ord('s'):
                status = "enabled" if self.email_notifier.is_emails_enabled() else "disabled"
                print(f"Email notifications are currently {status}")

        cap.release()
        cv2.destroyAllWindows()
        print("Face detection stopped")

if __name__ == "__main__":
    detector = detection()
    detector.run()
