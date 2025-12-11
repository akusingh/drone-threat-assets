"""
Data models for the Multimodal Drone Threat Analyzer.

These models define the structure of data flowing through the system,
from sensor outputs to threat assessments.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional


@dataclass
class RFDetection:
    """RF signal detection result from CNN model."""
    score: float  # 0-100
    frequency: float  # Hz
    bandwidth: float  # Hz
    modulation: str
    timestamp: datetime
    location: tuple  # (lat, lon)
    signal_strength: float  # dBm


@dataclass
class DetectedObject:
    """Individual object detected in visual frame."""
    bbox: tuple  # (x, y, w, h)
    classification: str  # drone, bird, aircraft, unknown
    altitude_estimate: float  # meters
    distance_estimate: float  # meters
    confidence: float  # 0-100


@dataclass
class VisualDetection:
    """Visual detection result from camera analysis."""
    objects: List[DetectedObject]
    confidence: float  # 0-100
    timestamp: datetime
    camera_id: str
    weather_conditions: str  # clear, fog, rain, etc.


@dataclass
class AudioDetection:
    """Audio detection result from microphone analysis."""
    score: float  # 0-100
    motor_count: int
    frequency_peaks: List[float]  # Hz
    timestamp: datetime
    microphone_id: str
    background_noise_level: float  # dB


@dataclass
class WeatherData:
    """Current weather conditions."""
    wind_speed: float  # m/s
    visibility: float  # meters
    conditions: str  # clear, cloudy, rain, fog
    temperature: float  # celsius


@dataclass
class FlightPermit:
    """Authorized flight permit."""
    permit_id: str
    operator: str
    valid_from: datetime
    valid_until: datetime
    area: tuple  # (lat, lon, radius_meters)
    drone_type: str


@dataclass
class LocalEvent:
    """Local event that might involve authorized drone use."""
    event_id: str
    name: str
    location: tuple  # (lat, lon)
    start_time: datetime
    end_time: datetime
    event_type: str  # wedding, film_shoot, construction, etc.


@dataclass
class AirspaceRestriction:
    """Airspace restriction or no-fly zone."""
    restriction_id: str
    area: tuple  # (lat, lon, radius_meters)
    altitude_limit: float  # meters
    restriction_type: str  # temporary, permanent
    reason: str


@dataclass
class ScheduledFlight:
    """Scheduled manned aircraft flight."""
    flight_id: str
    aircraft_type: str
    scheduled_time: datetime
    flight_path: List[tuple]  # List of (lat, lon) waypoints


@dataclass
class ContextData:
    """Contextual information for threat assessment."""
    weather: WeatherData
    permits: List[FlightPermit]
    events: List[LocalEvent]
    restrictions: List[AirspaceRestriction]
    flight_schedule: List[ScheduledFlight]
    timestamp: datetime
    location: tuple  # (lat, lon)


@dataclass
class ThreatAssessment:
    """Complete threat assessment from Gemini 3."""
    classification: str  # authorized, low, medium, high
    confidence: float  # 0-100
    explanation: str  # natural language reasoning
    recommended_actions: List[str]
    supporting_evidence: Dict[str, Any]
    timestamp: datetime
    assessment_id: str
    
    # Sensor inputs that led to this assessment
    rf_detection: Optional[RFDetection] = None
    visual_detection: Optional[VisualDetection] = None
    audio_detection: Optional[AudioDetection] = None
    context_data: Optional[ContextData] = None


@dataclass
class Feedback:
    """Operator feedback on a threat assessment."""
    assessment_id: str
    operator_id: str
    timestamp: datetime
    correct: bool  # Was the assessment correct?
    actual_classification: Optional[str]  # If incorrect, what should it have been?
    comments: str
    context: Dict[str, Any]  # Full context at time of feedback


def validate_threat_classification(classification: str) -> bool:
    """Validate that threat classification is one of the allowed values."""
    valid_classifications = {"authorized", "low", "medium", "high"}
    return classification in valid_classifications


def validate_confidence_score(score: float) -> bool:
    """Validate that confidence score is in valid range."""
    return 0.0 <= score <= 100.0
