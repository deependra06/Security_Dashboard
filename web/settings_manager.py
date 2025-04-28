import json
import os
from pathlib import Path

class SettingsManager:
    def __init__(self):
        self.settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'settings.json')
        self.default_settings = {
            # Camera settings
            'camera_id': 0,
            'frame_width': 640,
            'frame_height': 480,
            'fps': 10,
            
            # Email settings
            'email_enabled': False,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'smtp_username': '',
            'smtp_password': '',
            'sender_email': '',
            'recipient_email': '',
            
            # Storage settings
            'max_snapshots': 100,
            'max_events': 100,
            'auto_cleanup': False
        }
        self.settings = self.load_settings()

    def load_settings(self):
        """Load settings from file or return default settings if file doesn't exist"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    # Merge with default settings to ensure all keys exist
                    merged_settings = {**self.default_settings, **settings}
                    # Ensure numeric values are properly converted
                    for key in ['max_snapshots', 'max_events', 'frame_width', 'frame_height', 'fps', 'smtp_port']:
                        if key in merged_settings:
                            try:
                                merged_settings[key] = int(merged_settings[key])
                            except (ValueError, TypeError):
                                merged_settings[key] = self.default_settings[key]
                    # Ensure boolean values are properly converted
                    for key in ['email_enabled', 'auto_cleanup']:
                        if key in merged_settings:
                            merged_settings[key] = bool(merged_settings[key])
                    return merged_settings
            return self.default_settings.copy()
        except Exception as e:
            print(f"Error loading settings: {str(e)}")
            return self.default_settings.copy()

    def save_settings(self):
        """Save current settings to file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
            
            # Ensure all settings are properly serializable
            settings_to_save = self.settings.copy()
            for key, value in settings_to_save.items():
                if isinstance(value, (int, float, bool, str, list, dict)):
                    continue
                settings_to_save[key] = str(value)
            
            with open(self.settings_file, 'w') as f:
                json.dump(settings_to_save, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving settings: {str(e)}")
            return False

    def get_setting(self, key, default=None):
        """Get a specific setting value"""
        return self.settings.get(key, default)

    def update_setting(self, key, value):
        """Update a specific setting and save to file"""
        self.settings[key] = value
        return self.save_settings()

    def update_settings(self, settings_dict):
        """Update multiple settings at once and save to file"""
        self.settings.update(settings_dict)
        return self.save_settings()

    def reset_settings(self):
        """Reset all settings to default values"""
        self.settings = self.default_settings.copy()
        return self.save_settings() 