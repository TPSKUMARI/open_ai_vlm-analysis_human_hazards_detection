import cv2
import time
import base64
import sqlite3
import pytz
import json
import os
from datetime import datetime
from typing import Dict, Optional, List
import threading
import queue
import numpy as np
from dotenv import load_dotenv

# OpenAI imports
try:
    from openai import OpenAI
except ImportError:
    print("OpenAI library not installed. Install with: pip install openai")
    exit(1)

load_dotenv()  # reads OPENAI_API_KEY (and anything else) from a local .env file

class EnhancedVLMAnalysisSystem:
    """Enhanced Vision Language Model Analysis System with improved safety detection"""
    
    def __init__(self, api_key: str, analysis_interval=60, show_gui=False, save_frames=True, frames_folder="vlm_frames"):
        # api_key is expected to come from the caller's environment (OPENAI_API_KEY),
        # never hardcoded - see README for .env setup.
        self.api_key = api_key
        self.analysis_interval = analysis_interval
        self.show_gui = show_gui
        self.save_frames = save_frames
        self.frames_folder = frames_folder
        
        # Create frames folder if saving is enabled
        if self.save_frames:
            self.setup_frames_folder()
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize database
        self.init_database()
        
        # Timing control
        self.last_analysis_time = 0
        self.frame_count = 0
        
        # Enhanced status tracking
        self.current_status = {
            'human_activity': 'No analysis yet',
            'hazard_detected': False,
            'hazard_description': 'No hazards detected',
            'pre_fall_risk': False,
            'pre_fall_description': 'No pre-fall risks detected',
            'environmental_hazards': [],
            'safety_score': 100,
            'urgency_level': 'low',  # low, medium, high, critical
            'analysis_confidence': 0.0,
            'last_analysis_time': None,
            'frames_processed': 0,
            'api_calls_made': 0,
            'frames_saved': 0,
            'safety_recommendations': []
        }
        
        # Analysis queue for processing
        self.analysis_queue = queue.Queue()
        self.processing_thread = None
        self.running = False
        
        print("Enhanced VLM Analysis System initialized")
        if self.save_frames:
            print(f"Frame saving enabled - folder: {self.frames_folder}")
    
    def setup_frames_folder(self):
        """Create folder structure for saving frames"""
        try:
            if not os.path.exists(self.frames_folder):
                os.makedirs(self.frames_folder)
                print(f"Created frames folder: {self.frames_folder}")
            
            date_folder = os.path.join(self.frames_folder, datetime.now().strftime('%Y-%m-%d'))
            if not os.path.exists(date_folder):
                os.makedirs(date_folder)
                print(f"Created date folder: {date_folder}")
                
        except Exception as e:
            print(f"Error creating frames folder: {e}")
            self.save_frames = False
    
    def init_database(self):
        """Initialize enhanced database for VLM analysis logs"""
        try:
            self.conn = sqlite3.connect('enhanced_vlm_analysis_logs.db', check_same_thread=False)
            self.cursor = self.conn.cursor()
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS enhanced_vlm_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    human_activity TEXT,
                    hazard_detected BOOLEAN,
                    hazard_description TEXT,
                    pre_fall_risk BOOLEAN,
                    pre_fall_description TEXT,
                    environmental_hazards TEXT,
                    safety_score INTEGER,
                    urgency_level TEXT,
                    emotional_state TEXT,
                    confidence REAL,
                    safety_recommendations TEXT,
                    additional_notes TEXT,
                    raw_response TEXT,
                    api_call_success BOOLEAN,
                    processing_time REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.conn.commit()
            print("Enhanced VLM analysis database initialized")
        except Exception as e:
            print(f"Database initialization error: {e}")
    
    def encode_frame_to_base64(self, frame):
        """Convert frame to base64 for OpenAI API"""
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            base64_image = base64.b64encode(buffer).decode('utf-8')
            return base64_image
        except Exception as e:
            print(f"Error encoding frame: {e}")
            return None
    
    def analyze_frame_with_enhanced_openai(self, base64_image):
        """Enhanced frame analysis with comprehensive safety detection"""
        start_time = time.time()
        api_success = False
        
        try:
            # Enhanced prompt for comprehensive safety analysis
            prompt = """
            You are an advanced AI safety monitor for elderly care. Analyze this image for comprehensive safety assessment.

            CRITICAL ANALYSIS AREAS:

            1. HUMAN ACTIVITY:
            - Describe current activity in detail (sitting, standing, walking, cooking, cleaning, etc.)
            - Assess if the activity is being performed safely
            - Note posture and body positioning

            2. PRE-FALL RISK ASSESSMENT:
            - Look for signs that indicate potential falling risk:
              * Unsteady posture or balance issues
              * Reaching for objects while off-balance
              * Standing on unstable surfaces (chairs, stools)
              * Poor lighting conditions
              * Rushing or moving too quickly
              * Signs of dizziness or disorientation
              * Leaning too far forward or backward
              * One foot raised or unstable stance
            
            3. ENVIRONMENTAL HAZARDS:
            - Spilled liquids on floor (water, drinks, cleaning products)
            - Scattered objects or obstacles in walkways
            - Loose rugs or mats
            - Poor lighting or shadows
            - Sharp objects within reach
            - Stairs without proper railings
            - Wet surfaces (bathrooms, kitchens)
            - Electrical hazards (exposed wires, overloaded outlets)
            - Fire/smoke/gas hazards
            - Open windows or doors creating safety risks
            - Cluttered pathways

            4. SAFETY EQUIPMENT ASSESSMENT:
            - Is person wearing appropriate safety gear if needed?
            - Are mobility aids (walker, cane) being used properly?
            - Safety railings or grab bars available and being used?

            5. HEALTH & BEHAVIORAL INDICATORS:
            - Signs of confusion or disorientation?
            - Physical distress indicators?
            - Unusual or risky behavior patterns?

            Calculate SAFETY SCORE (0-100):
            - 90-100: Very safe environment and behavior
            - 70-89: Generally safe with minor concerns
            - 50-69: Moderate safety concerns requiring attention
            - 30-49: Significant safety risks present
            - 0-29: Critical safety situation requiring immediate intervention

            Determine URGENCY LEVEL:
            - "low": Normal activities, safe environment
            - "medium": Minor hazards or slightly risky behavior
            - "high": Significant risks that need prompt attention
            - "critical": Immediate danger requiring emergency response

            Respond in this exact JSON format:
            {
                "human_activity": "detailed description of current activity and how it's being performed",
                "hazard_detected": true/false,
                "hazard_description": "specific description of immediate hazards or 'No immediate hazards detected'",
                "pre_fall_risk": true/false,
                "pre_fall_description": "detailed assessment of fall risks or 'No pre-fall risks detected'",
                "environmental_hazards": ["list", "of", "environmental", "concerns"],
                "safety_score": 85,
                "urgency_level": "low/medium/high/critical",
                "emotional_state": "calm/anxious/confused/distressed/content",
                "confidence": 0.9,
                "additional_notes": "any other relevant safety observations",
                "safety_recommendations": ["specific", "safety", "suggestions"]
            }
            """
            
            # Make API call to OpenAI with enhanced model
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Use more powerful model for better accuracy
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=800,  # Increased for more detailed analysis
                temperature=0.1
            )
            
            processing_time = time.time() - start_time
            api_success = True
            
            raw_response = response.choices[0].message.content
            
            # Parse JSON response
            try:
                json_start = raw_response.find('{')
                json_end = raw_response.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_str = raw_response[json_start:json_end]
                    analysis_result = json.loads(json_str)
                else:
                    analysis_result = self.create_fallback_response(raw_response)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                analysis_result = self.create_fallback_response(raw_response)
            
            # Add metadata
            analysis_result['raw_response'] = raw_response
            analysis_result['api_call_success'] = api_success
            analysis_result['processing_time'] = processing_time
            
            return analysis_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            print(f"OpenAI API error: {error_msg}")
            
            return {
                "human_activity": f"API Error: {error_msg[:50]}",
                "hazard_detected": False,
                "hazard_description": "API call failed - unable to assess hazards",
                "pre_fall_risk": False,
                "pre_fall_description": "API call failed - unable to assess fall risk",
                "environmental_hazards": [],
                "safety_score": 0,
                "urgency_level": "unknown",
                "emotional_state": "Unknown",
                "confidence": 0.0,
                "additional_notes": f"Error: {error_msg}",
                "safety_recommendations": [],
                "raw_response": f"API Error: {error_msg}",
                "api_call_success": False,
                "processing_time": processing_time
            }
    
    def create_fallback_response(self, raw_response):
        """Create fallback response when JSON parsing fails"""
        return {
            "human_activity": "Unable to parse activity from response",
            "hazard_detected": False,
            "hazard_description": "Unable to assess hazards",
            "pre_fall_risk": False,
            "pre_fall_description": "Unable to assess fall risk",
            "environmental_hazards": [],
            "safety_score": 50,
            "urgency_level": "unknown",
            "emotional_state": "Unknown",
            "confidence": 0.0,
            "additional_notes": raw_response[:200] if raw_response else "No response",
            "safety_recommendations": []
        }
    
    def process_analysis_queue(self):
        """Background thread to process analysis requests"""
        while self.running:
            try:
                frame_data = self.analysis_queue.get(timeout=1)
                if frame_data is None:
                    continue
                
                frame = frame_data['frame']
                timestamp = frame_data['timestamp']
                
                print(f"Processing Enhanced VLM analysis for frame at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}")
                
                # Save frame to disk BEFORE analysis
                saved_filepath = None
                if self.save_frames:
                    saved_filepath = self.save_frame_to_disk(frame)
                
                # Encode frame for OpenAI
                base64_image = self.encode_frame_to_base64(frame)
                if base64_image is None:
                    continue
                
                # Enhanced analysis with OpenAI
                analysis_result = self.analyze_frame_with_enhanced_openai(base64_image)
                
                # Update status with enhanced information
                self.current_status.update({
                    'human_activity': analysis_result.get('human_activity', 'Unknown activity'),
                    'hazard_detected': analysis_result.get('hazard_detected', False),
                    'hazard_description': analysis_result.get('hazard_description', 'No assessment available'),
                    'pre_fall_risk': analysis_result.get('pre_fall_risk', False),
                    'pre_fall_description': analysis_result.get('pre_fall_description', 'No pre-fall assessment'),
                    'environmental_hazards': analysis_result.get('environmental_hazards', []),
                    'safety_score': analysis_result.get('safety_score', 50),
                    'urgency_level': analysis_result.get('urgency_level', 'unknown'),
                    'analysis_confidence': analysis_result.get('confidence', 0.0),
                    'last_analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'api_calls_made': self.current_status['api_calls_made'] + 1,
                    'safety_recommendations': analysis_result.get('safety_recommendations', [])
                })
                
                # Save analysis metadata
                if saved_filepath:
                    self.save_analysis_metadata(saved_filepath, analysis_result)
                
                # Log to database
                self.log_enhanced_analysis(analysis_result)
                
                # Print comprehensive analysis result
                self.print_analysis_result(analysis_result)
                
                self.analysis_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in analysis processing: {e}")
    
    def print_analysis_result(self, analysis_result):
        """Print comprehensive analysis results"""
        print(f"\n{'='*60}")
        print("ENHANCED VLM ANALYSIS RESULT")
        print(f"{'='*60}")
        print(f"Activity: {analysis_result.get('human_activity', 'Unknown')}")
        print(f"Safety Score: {analysis_result.get('safety_score', 0)}/100")
        print(f"Urgency Level: {analysis_result.get('urgency_level', 'unknown').upper()}")
        
        if analysis_result.get('hazard_detected', False):
            print(f"WARNING HAZARD: {analysis_result.get('hazard_description', 'Unknown hazard')}")
        
        if analysis_result.get('pre_fall_risk', False):
            print(f"PRE-FALL RISK: {analysis_result.get('pre_fall_description', 'Unknown risk')}")
        
        env_hazards = analysis_result.get('environmental_hazards', [])
        if env_hazards:
            print(f"Environmental Hazards: {', '.join(env_hazards)}")
        
        print(f"Emotional State: {analysis_result.get('emotional_state', 'Unknown')}")
        print(f"Confidence: {analysis_result.get('confidence', 0.0):.2f}")
        print(f"Processing Time: {analysis_result.get('processing_time', 0.0):.2f}s")
        
        # Safety recommendations
        recommendations = analysis_result.get('safety_recommendations', [])
        if recommendations:
            print(f"Safety Recommendations:")
            for rec in recommendations[:3]:  # Show top 3
                print(f"   - {rec}")
        
        print(f"{'='*60}\n")
    
    def save_frame_to_disk(self, frame, analysis_result=None):
        """Save frame to disk with timestamp and analysis info"""
        if not self.save_frames:
            return None
            
        try:
            current_time = datetime.now()
            date_str = current_time.strftime('%Y-%m-%d')
            time_str = current_time.strftime('%H-%M-%S')
            
            date_folder = os.path.join(self.frames_folder, date_str)
            if not os.path.exists(date_folder):
                os.makedirs(date_folder)
            
            filename = f"{date_str}_{time_str}_frame_{self.current_status['api_calls_made']:04d}.jpg"
            filepath = os.path.join(date_folder, filename)
            
            success = cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if success:
                self.current_status['frames_saved'] += 1
                print(f"Frame saved: {filename}")
                return filepath
            else:
                print(f"Failed to save frame: {filename}")
                return None
                
        except Exception as e:
            print(f"Error saving frame: {e}")
            return None
    
    def save_analysis_metadata(self, image_filepath, analysis_result):
        """Save enhanced analysis metadata"""
        try:
            base_name = os.path.splitext(image_filepath)[0]
            metadata_filepath = f"{base_name}_metadata.json"
            
            metadata = {
                'image_file': os.path.basename(image_filepath),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'analysis': {
                    'human_activity': analysis_result.get('human_activity', ''),
                    'hazard_detected': analysis_result.get('hazard_detected', False),
                    'hazard_description': analysis_result.get('hazard_description', ''),
                    'pre_fall_risk': analysis_result.get('pre_fall_risk', False),
                    'pre_fall_description': analysis_result.get('pre_fall_description', ''),
                    'environmental_hazards': analysis_result.get('environmental_hazards', []),
                    'safety_score': analysis_result.get('safety_score', 50),
                    'urgency_level': analysis_result.get('urgency_level', 'unknown'),
                    'emotional_state': analysis_result.get('emotional_state', ''),
                    'confidence': analysis_result.get('confidence', 0.0),
                    'safety_recommendations': analysis_result.get('safety_recommendations', []),
                    'api_call_success': analysis_result.get('api_call_success', False),
                    'processing_time': analysis_result.get('processing_time', 0.0)
                }
            }
            
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            print(f"Enhanced metadata saved: {os.path.basename(metadata_filepath)}")
            
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def log_enhanced_analysis(self, analysis_result):
        """Log enhanced analysis result to database"""
        try:
            slst_tz = pytz.timezone('Asia/Colombo')
            current_time = datetime.now(slst_tz).strftime('%Y-%m-%d %H:%M:%S')
            
            self.cursor.execute('''
                INSERT INTO enhanced_vlm_logs 
                (human_activity, hazard_detected, hazard_description, pre_fall_risk, 
                 pre_fall_description, environmental_hazards, safety_score, urgency_level,
                 emotional_state, confidence, safety_recommendations, additional_notes, 
                 raw_response, api_call_success, processing_time, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis_result.get('human_activity', ''),
                analysis_result.get('hazard_detected', False),
                analysis_result.get('hazard_description', ''),
                analysis_result.get('pre_fall_risk', False),
                analysis_result.get('pre_fall_description', ''),
                json.dumps(analysis_result.get('environmental_hazards', [])),
                analysis_result.get('safety_score', 50),
                analysis_result.get('urgency_level', 'unknown'),
                analysis_result.get('emotional_state', ''),
                analysis_result.get('confidence', 0.0),
                json.dumps(analysis_result.get('safety_recommendations', [])),
                analysis_result.get('additional_notes', ''),
                analysis_result.get('raw_response', ''),
                analysis_result.get('api_call_success', False),
                analysis_result.get('processing_time', 0.0),
                current_time
            ))
            self.conn.commit()
            
            print(f"Enhanced VLM analysis logged to database successfully")
            
        except Exception as e:
            print(f"Enhanced VLM logging error: {e}")
    
    def process_frame(self, frame, human_detected=False):
        """Process frame with enhanced analysis"""
        current_time = time.time()
        self.frame_count += 1
        
        self.current_status['frames_processed'] = self.frame_count
        
        if not human_detected:
            time_until_next = max(0, self.analysis_interval - (current_time - self.last_analysis_time))
            
            return {
                'human_activity': 'No human detected - VLM analysis paused',
                'hazard_detected': False,
                'hazard_description': 'No analysis - waiting for human detection',
                'pre_fall_risk': False,
                'pre_fall_description': 'No analysis - waiting for human detection',
                'environmental_hazards': [],
                'safety_score': 100,
                'urgency_level': 'low',
                'analysis_confidence': 0.0,
                'last_analysis_time': self.current_status['last_analysis_time'],
                'time_until_next_analysis': time_until_next
            }
        
        if current_time - self.last_analysis_time >= self.analysis_interval:
            print(f"\nProcessing Enhanced VLM analysis for Frame {self.frame_count} (Human detected)...")
            
            # Clear queue and add current frame
            while not self.analysis_queue.empty():
                try:
                    self.analysis_queue.get_nowait()
                    self.analysis_queue.task_done()
                except queue.Empty:
                    break
            
            frame_data = {
                'frame': frame.copy(),
                'timestamp': current_time,
                'human_detected': True
            }
            
            try:
                self.analysis_queue.put_nowait(frame_data)
                self.last_analysis_time = current_time
                print(f"Enhanced VLM analysis queued - Next analysis in {self.analysis_interval} seconds")
            except queue.Full:
                print("Analysis queue is full, clearing and adding current frame")
                while not self.analysis_queue.empty():
                    try:
                        self.analysis_queue.get_nowait()
                        self.analysis_queue.task_done()
                    except queue.Empty:
                        break
                self.analysis_queue.put_nowait(frame_data)
                self.last_analysis_time = current_time
        
        time_until_next = max(0, self.analysis_interval - (current_time - self.last_analysis_time))
        
        return {
            'human_activity': self.current_status['human_activity'],
            'hazard_detected': self.current_status['hazard_detected'],
            'hazard_description': self.current_status['hazard_description'],
            'pre_fall_risk': self.current_status['pre_fall_risk'],
            'pre_fall_description': self.current_status['pre_fall_description'],
            'environmental_hazards': self.current_status['environmental_hazards'],
            'safety_score': self.current_status['safety_score'],
            'urgency_level': self.current_status['urgency_level'],
            'analysis_confidence': self.current_status['analysis_confidence'],
            'last_analysis_time': self.current_status['last_analysis_time'],
            'time_until_next_analysis': time_until_next
        }
    
    def start_processing(self):
        """Start the background processing thread"""
        self.running = True
        self.processing_thread = threading.Thread(target=self.process_analysis_queue, daemon=True)
        self.processing_thread.start()
        print("Enhanced VLM processing thread started")
    
    def stop_processing(self):
        """Stop the background processing thread"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        print("Enhanced VLM processing thread stopped")
    
    def get_status(self):
        """Get current enhanced system status"""
        return self.current_status.copy()
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.stop_processing()
            if hasattr(self, 'conn'):
                self.conn.close()
            print("Enhanced VLM Analysis System cleaned up")
        except Exception as e:
            print(f"Enhanced VLM cleanup error: {e}")

# For standalone testing
if __name__ == "__main__":
    print("Testing Enhanced VLM Analysis System standalone...")

    # API key is pulled from the environment (.env) rather than hardcoded here -
    # see .env.example for the variable name.
    API_KEY = os.getenv("OPENAI_API_KEY")

    if not API_KEY:
        print("Please set OPENAI_API_KEY in your .env file before testing")
        exit(1)
    
    # Test with camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No camera available for testing")
        exit(1)
    
    # Create system with shorter interval for testing (10 seconds instead of 60)
    system = EnhancedVLMAnalysisSystem(api_key=API_KEY, analysis_interval=10, show_gui=True)
    system.start_processing()
    
    try:
        print("Starting Enhanced VLM analysis test...")
        print("Frame will be analyzed every 10 seconds")
        print("Controls: Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Mirror frame for better user experience
            frame = cv2.flip(frame, 1)
            
            # Process frame
            result = system.process_frame(frame, human_detected=True)  # Assume human detected for testing
            
            # Display frame with analysis info
            cv2.putText(frame, f"Activity: {result['human_activity'][:30]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Safety Score: {result['safety_score']}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Pre-Fall Risk: {result['pre_fall_risk']}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0) if result['pre_fall_risk'] else (0, 255, 0), 2)
            
            cv2.imshow("Enhanced VLM Analysis Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        system.cleanup()
        cv2.destroyAllWindows()
        print("Enhanced VLM Analysis System test complete")


        