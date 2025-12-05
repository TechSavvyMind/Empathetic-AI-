"""
sentiment_analyzer.py - Advanced 27-Emotion Detection System
-------------------------------------------------------------
Uses state-of-the-art HuggingFace models fine-tuned for detecting
27 distinct human emotions with intensity scoring.

Primary Model: SamLowe/roberta-base-go_emotions (28 emotions)
Fallback: j-hartmann/emotion-english-distilroberta-base (7 emotions)

Emotions Detected:
1. admiration          15. nervousness
2. amusement           16. neutral
3. anger               17. optimism
4. annoyance           18. pride
5. approval            19. realization
6. caring              20. relief
7. confusion           21. remorse
8. curiosity           22. sadness
9. desire              23. surprise
10. disappointment     24. gratitude
11. disapproval        25. grief
12. disgust            26. joy
13. embarrassment      27. love
14. excitement         28. fear
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import warnings

warnings.filterwarnings('ignore')

# Try importing transformers
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ transformers not installed. Install with: pip install transformers torch")


# ============================================================================
# EMOTION MODELS & CONFIGURATION
# ============================================================================

@dataclass
class EmotionScore:
    """Single emotion with score"""
    emotion: str
    score: float
    intensity: str  # low, medium, high, very_high
    
    def __post_init__(self):
        """Calculate intensity from score"""
        if self.score < 0.3:
            self.intensity = "low"
        elif self.score < 0.6:
            self.intensity = "medium"
        elif self.score < 0.8:
            self.intensity = "high"
        else:
            self.intensity = "very_high"


@dataclass
class SentimentAnalysisResult:
    """Complete sentiment analysis result"""
    text: str
    primary_emotion: EmotionScore
    top_emotions: List[EmotionScore]
    all_emotions: Dict[str, float]
    
    # Aggregated metrics
    overall_sentiment: str  # positive, negative, neutral, mixed
    sentiment_intensity: float  # 0-1
    
    # Contextual analysis
    is_urgent: bool
    is_frustrated: bool
    is_satisfied: bool
    needs_empathy: bool
    escalation_recommended: bool
    
    # Metadata
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "text": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "primary_emotion": {
                "emotion": self.primary_emotion.emotion,
                "score": round(self.primary_emotion.score, 3),
                "intensity": self.primary_emotion.intensity
            },
            "top_emotions": [
                {
                    "emotion": e.emotion,
                    "score": round(e.score, 3),
                    "intensity": e.intensity
                }
                for e in self.top_emotions
            ],
            "overall_sentiment": self.overall_sentiment,
            "sentiment_intensity": round(self.sentiment_intensity, 3),
            "is_urgent": self.is_urgent,
            "is_frustrated": self.is_frustrated,
            "needs_empathy": self.needs_empathy,
            "escalation_recommended": self.escalation_recommended,
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp.isoformat()
        }


# ============================================================================
# EMOTION CATEGORIZATION
# ============================================================================

# Map emotions to sentiment categories
EMOTION_CATEGORIES = {
    # Positive emotions
    "positive": [
        "admiration", "amusement", "approval", "caring", "desire",
        "excitement", "gratitude", "joy", "love", "optimism",
        "pride", "relief"
    ],
    
    # Negative emotions
    "negative": [
        "anger", "annoyance", "disappointment", "disapproval", "disgust",
        "embarrassment", "fear", "grief", "nervousness", "remorse",
        "sadness"
    ],
    
    # Neutral emotions
    "neutral": [
        "confusion", "curiosity", "neutral", "realization", "surprise"
    ],
    
    # High priority emotions (need immediate attention)
    "high_priority": [
        "anger", "fear", "grief", "disgust"
    ],
    
    # Frustration indicators
    "frustration": [
        "anger", "annoyance", "disappointment", "disapproval"
    ],
    
    # Satisfaction indicators
    "satisfaction": [
        "gratitude", "joy", "relief", "approval", "admiration"
    ]
}


# ============================================================================
# ADVANCED SENTIMENT ANALYZER
# ============================================================================

class AdvancedSentimentAnalyzer:
    """
    Production-grade sentiment analyzer with 27+ emotion detection
    """
    
    def __init__(
        self,
        model_name: str = "SamLowe/roberta-base-go_emotions",
        device: str = None,
        cache_size: int = 1000
    ):
        """
        Initialize sentiment analyzer
        
        Args:
            model_name: HuggingFace model to use
            device: 'cuda', 'cpu', or None (auto-detect)
            cache_size: Size of emotion history cache
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required. Install with: pip install transformers torch")
        
        self.model_name = model_name
        
        # Setup device
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = 0 if device == "cuda" else -1
        
        print(f"🔧 Initializing Sentiment Analyzer...")
        print(f"   Model: {model_name}")
        print(f"   Device: {'GPU' if self.device == 0 else 'CPU'}")
        
        # Load model and tokenizer
        try:
            self.classifier = pipeline(
                "text-classification",
                model=model_name,
                top_k=None,  # Return all emotions
                device=self.device
            )
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load primary model: {e}")
            print("   Falling back to secondary model...")
            self.classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,
                device=self.device
            )
            print("✅ Fallback model loaded")
        
        # Emotion history for trend analysis
        self.emotion_history = deque(maxlen=cache_size)
        
        # Statistics
        self.analysis_count = 0
        
    def analyze(
        self,
        text: str,
        return_all_emotions: bool = True,
        top_k: int = 5
    ) -> SentimentAnalysisResult:
        """
        Analyze text and return comprehensive sentiment analysis
        
        Args:
            text: Input text to analyze
            return_all_emotions: Whether to return all emotion scores
            top_k: Number of top emotions to return
        
        Returns:
            SentimentAnalysisResult with complete analysis
        """
        if not text or not text.strip():
            return self._create_neutral_result(text)
        
        # Run emotion detection
        try:
            raw_results = self.classifier(text)[0]
        except Exception as e:
            print(f"⚠️ Analysis failed: {e}")
            return self._create_neutral_result(text)
        
        # Parse results
        all_emotions = {item['label']: item['score'] for item in raw_results}
        
        # Sort by score
        sorted_emotions = sorted(
            all_emotions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create emotion objects
        primary_emotion = EmotionScore(
            emotion=sorted_emotions[0][0],
            score=sorted_emotions[0][1]
        )
        
        top_emotions = [
            EmotionScore(emotion=emotion, score=score)
            for emotion, score in sorted_emotions[:top_k]
        ]
        
        # Calculate overall sentiment
        overall_sentiment = self._calculate_overall_sentiment(all_emotions)
        sentiment_intensity = self._calculate_sentiment_intensity(all_emotions)
        
        # Contextual analysis
        is_urgent = self._detect_urgency(text, all_emotions)
        is_frustrated = self._detect_frustration(all_emotions)
        is_satisfied = self._detect_satisfaction(all_emotions)
        needs_empathy = self._assess_empathy_need(all_emotions, primary_emotion)
        escalation_recommended = self._assess_escalation(
            all_emotions,
            primary_emotion,
            is_urgent,
            is_frustrated
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(all_emotions, primary_emotion)
        
        # Create result
        result = SentimentAnalysisResult(
            text=text,
            primary_emotion=primary_emotion,
            top_emotions=top_emotions,
            all_emotions=all_emotions if return_all_emotions else {},
            overall_sentiment=overall_sentiment,
            sentiment_intensity=sentiment_intensity,
            is_urgent=is_urgent,
            is_frustrated=is_frustrated,
            is_satisfied=is_satisfied,
            needs_empathy=needs_empathy,
            escalation_recommended=escalation_recommended,
            confidence=confidence
        )
        
        # Update history
        self.emotion_history.append(result)
        self.analysis_count += 1
        
        return result
    
    def analyze_conversation_trend(
        self,
        recent_messages: List[str] = None
    ) -> Dict[str, any]:
        """
        Analyze emotion trends across conversation
        
        Args:
            recent_messages: List of recent messages to analyze (optional)
                           If None, uses cached history
        
        Returns:
            Dictionary with trend analysis
        """
        if recent_messages:
            # Analyze new messages
            analyses = [self.analyze(msg) for msg in recent_messages]
        else:
            # Use cached history
            analyses = list(self.emotion_history)
        
        if not analyses:
            return {"trend": "insufficient_data"}
        
        # Extract primary emotions over time
        emotion_sequence = [a.primary_emotion.emotion for a in analyses]
        intensity_sequence = [a.sentiment_intensity for a in analyses]
        
        # Detect trends
        is_escalating = self._detect_escalation_trend(intensity_sequence, emotion_sequence)
        is_improving = self._detect_improvement_trend(intensity_sequence, emotion_sequence)
        is_volatile = self._detect_volatility(emotion_sequence)
        
        # Calculate average sentiment
        avg_intensity = np.mean(intensity_sequence)
        
        # Most common emotions
        emotion_counts = {}
        for emotion in emotion_sequence:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        most_common = sorted(
            emotion_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return {
            "trend": "escalating" if is_escalating else "improving" if is_improving else "stable",
            "is_volatile": is_volatile,
            "average_intensity": round(avg_intensity, 3),
            "message_count": len(analyses),
            "most_common_emotions": [{"emotion": e, "count": c} for e, c in most_common],
            "current_emotion": analyses[-1].primary_emotion.emotion,
            "requires_intervention": is_escalating or (avg_intensity > 0.7 and emotion_sequence[-1] in EMOTION_CATEGORIES["frustration"])
        }
    
    def _calculate_overall_sentiment(self, emotions: Dict[str, float]) -> str:
        """Calculate overall sentiment category"""
        positive_score = sum(
            score for emotion, score in emotions.items()
            if emotion in EMOTION_CATEGORIES["positive"]
        )
        negative_score = sum(
            score for emotion, score in emotions.items()
            if emotion in EMOTION_CATEGORIES["negative"]
        )
        neutral_score = sum(
            score for emotion, score in emotions.items()
            if emotion in EMOTION_CATEGORIES["neutral"]
        )
        
        scores = {
            "positive": positive_score,
            "negative": negative_score,
            "neutral": neutral_score
        }
        
        max_category = max(scores.items(), key=lambda x: x[1])
        
        # Check for mixed sentiment
        if max_category[1] < 0.4 or abs(positive_score - negative_score) < 0.2:
            return "mixed"
        
        return max_category[0]
    
    def _calculate_sentiment_intensity(self, emotions: Dict[str, float]) -> float:
        """Calculate overall intensity of sentiment"""
        # Use top 3 emotions
        top_scores = sorted(emotions.values(), reverse=True)[:3]
        return sum(top_scores) / 3 if top_scores else 0.0
    
    def _detect_urgency(self, text: str, emotions: Dict[str, float]) -> bool:
        """Detect if message is urgent"""
        urgent_keywords = [
            'urgent', 'immediately', 'asap', 'emergency', 'critical',
            'now', 'right now', 'right away'
        ]
        
        has_urgent_keyword = any(kw in text.lower() for kw in urgent_keywords)
        
        # High anger/fear also indicates urgency
        high_priority_emotions = sum(
            score for emotion, score in emotions.items()
            if emotion in EMOTION_CATEGORIES["high_priority"] and score > 0.6
        )
        
        return has_urgent_keyword or high_priority_emotions > 0.5
    
    def _detect_frustration(self, emotions: Dict[str, float]) -> bool:
        """Detect frustration"""
        frustration_score = sum(
            score for emotion, score in emotions.items()
            if emotion in EMOTION_CATEGORIES["frustration"]
        )
        return frustration_score > 0.5
    
    def _detect_satisfaction(self, emotions: Dict[str, float]) -> bool:
        """Detect satisfaction"""
        satisfaction_score = sum(
            score for emotion, score in emotions.items()
            if emotion in EMOTION_CATEGORIES["satisfaction"]
        )
        return satisfaction_score > 0.5
    
    def _assess_empathy_need(
        self,
        emotions: Dict[str, float],
        primary_emotion: EmotionScore
    ) -> bool:
        """Determine if response needs empathy"""
        # Negative emotions with high intensity need empathy
        if primary_emotion.emotion in EMOTION_CATEGORIES["negative"]:
            if primary_emotion.score > 0.5:
                return True
        
        # High priority emotions always need empathy
        if primary_emotion.emotion in EMOTION_CATEGORIES["high_priority"]:
            return True
        
        return False
    
    def _assess_escalation(
        self,
        emotions: Dict[str, float],
        primary_emotion: EmotionScore,
        is_urgent: bool,
        is_frustrated: bool
    ) -> bool:
        """Determine if escalation to human is recommended"""
        # Critical emotions
        if primary_emotion.emotion in ['anger', 'grief'] and primary_emotion.score > 0.7:
            return True
        
        # Urgent + frustrated
        if is_urgent and is_frustrated:
            return True
        
        # Very high intensity negative emotion
        if (primary_emotion.emotion in EMOTION_CATEGORIES["negative"] and
            primary_emotion.intensity == "very_high"):
            return True
        
        return False
    
    def _calculate_confidence(
        self,
        emotions: Dict[str, float],
        primary_emotion: EmotionScore
    ) -> float:
        """Calculate confidence in analysis"""
        # High primary score = high confidence
        if primary_emotion.score > 0.8:
            return 0.95
        
        # Check if primary is significantly higher than second
        sorted_scores = sorted(emotions.values(), reverse=True)
        if len(sorted_scores) >= 2:
            gap = sorted_scores[0] - sorted_scores[1]
            if gap > 0.3:
                return 0.9
            elif gap > 0.2:
                return 0.8
            else:
                return 0.7
        
        return 0.75
    
    def _detect_escalation_trend(
        self,
        intensity_sequence: List[float],
        emotion_sequence: List[str]
    ) -> bool:
        """Detect if emotions are escalating negatively"""
        if len(intensity_sequence) < 3:
            return False
        
        # Check if intensity is increasing
        recent_intensities = intensity_sequence[-3:]
        is_increasing = all(
            recent_intensities[i] < recent_intensities[i+1]
            for i in range(len(recent_intensities)-1)
        )
        
        # Check if moving toward negative emotions
        recent_emotions = emotion_sequence[-3:]
        becoming_negative = sum(
            1 for e in recent_emotions
            if e in EMOTION_CATEGORIES["negative"]
        ) >= 2
        
        return is_increasing and becoming_negative
    
    def _detect_improvement_trend(
        self,
        intensity_sequence: List[float],
        emotion_sequence: List[str]
    ) -> bool:
        """Detect if emotions are improving"""
        if len(emotion_sequence) < 3:
            return False
        
        # Check if moving toward positive emotions
        recent_emotions = emotion_sequence[-3:]
        becoming_positive = sum(
            1 for e in recent_emotions
            if e in EMOTION_CATEGORIES["positive"]
        ) >= 2
        
        return becoming_positive
    
    def _detect_volatility(self, emotion_sequence: List[str]) -> bool:
        """Detect if emotions are volatile (changing rapidly)"""
        if len(emotion_sequence) < 4:
            return False
        
        # Count unique emotions in recent history
        unique_emotions = len(set(emotion_sequence[-5:]))
        return unique_emotions >= 4
    
    def _create_neutral_result(self, text: str) -> SentimentAnalysisResult:
        """Create neutral result for edge cases"""
        neutral_emotion = EmotionScore(emotion="neutral", score=1.0)
        
        return SentimentAnalysisResult(
            text=text,
            primary_emotion=neutral_emotion,
            top_emotions=[neutral_emotion],
            all_emotions={"neutral": 1.0},
            overall_sentiment="neutral",
            sentiment_intensity=0.0,
            is_urgent=False,
            is_frustrated=False,
            is_satisfied=False,
            needs_empathy=False,
            escalation_recommended=False,
            confidence=1.0
        )
    
    def get_statistics(self) -> Dict:
        """Get analyzer statistics"""
        return {
            "total_analyses": self.analysis_count,
            "history_size": len(self.emotion_history),
            "model": self.model_name,
            "device": "GPU" if self.device == 0 else "CPU"
        }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def main():
    """Example usage"""
    print("🚀 Initializing Advanced Sentiment Analyzer\n")
    
    # Initialize analyzer
    analyzer = AdvancedSentimentAnalyzer()
    
    # Test cases
    test_cases = [
        "I'm so happy with your service! Everything works perfectly.",
        "This is absolutely terrible. I've been waiting for 3 days!",
        "I'm not sure what to do. Can you help me understand this?",
        "I AM FURIOUS! This is the third time I'm calling about the same issue!",
        "Thank you so much for resolving this quickly. I really appreciate it.",
        "My internet is slow but I'm paying for the premium plan. What's going on?",
    ]
    
    print("="*80)
    print("SENTIMENT ANALYSIS RESULTS")
    print("="*80 + "\n")
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n{'─'*80}")
        print(f"Test Case {i}")
        print(f"{'─'*80}")
        print(f"Text: \"{text}\"\n")
        
        # Analyze
        result = analyzer.analyze(text)
        
        # Print results
        print(f"Primary Emotion: {result.primary_emotion.emotion.upper()}")
        print(f"  Score: {result.primary_emotion.score:.3f}")
        print(f"  Intensity: {result.primary_emotion.intensity}")
        
        print(f"\nTop 3 Emotions:")
        for j, emotion in enumerate(result.top_emotions[:3], 1):
            print(f"  {j}. {emotion.emotion}: {emotion.score:.3f} ({emotion.intensity})")
        
        print(f"\nOverall Sentiment: {result.overall_sentiment.upper()}")
        print(f"Sentiment Intensity: {result.sentiment_intensity:.3f}")
        
        print(f"\nContextual Flags:")
        print(f"  🚨 Urgent: {result.is_urgent}")
        print(f"  😤 Frustrated: {result.is_frustrated}")
        print(f"  😊 Satisfied: {result.is_satisfied}")
        print(f"  💙 Needs Empathy: {result.needs_empathy}")
        print(f"  📞 Escalation Recommended: {result.escalation_recommended}")
        
        print(f"\n✅ Confidence: {result.confidence:.3f}")
    
    # Analyze conversation trend
    print(f"\n\n{'='*80}")
    print("CONVERSATION TREND ANALYSIS")
    print(f"{'='*80}\n")
    
    trend = analyzer.analyze_conversation_trend()
    print(f"Trend: {trend['trend'].upper()}")
    print(f"Volatile: {trend['is_volatile']}")
    print(f"Average Intensity: {trend['average_intensity']:.3f}")
    print(f"Requires Intervention: {trend['requires_intervention']}")
    print(f"\nMost Common Emotions:")
    for item in trend['most_common_emotions']:
        print(f"  - {item['emotion']}: {item['count']} times")
    
    # Print statistics
    print(f"\n\n{'='*80}")
    print("ANALYZER STATISTICS")
    print(f"{'='*80}\n")
    stats = analyzer.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()