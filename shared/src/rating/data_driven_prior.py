"""
Data Quality-Driven Prior Information System

This system accesses the quality of information used/loaded by the data management system
to provide prior knowledge based on data reliability, completeness, and validation metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import json

@dataclass
class DataQualityMetric:
    """Represents quality metrics for a piece of information"""
    metric_id: str
    description: str
    quality_score: float  # 0-1, how reliable/complete/valid this data is
    data_source: str  # Where this data came from
    completeness: float  # 0-1, how complete the data is
    validation_status: str  # 'validated', 'unvalidated', 'suspicious', 'corrupted'
    confidence_level: float  # 0-1, statistical confidence in the data
    information_gain_quality: float  # Quality of information gain measurements
    uncertainty_reliability: float  # How reliable the uncertainty measurements are
    timestamp: str  # When this quality assessment was made

class DataPointRatingSystem:
    """
    Prior information system that accesses and provides scores/ratings of individual data points.
    
    This system evaluates and rates individual data points to provide:
    - Quality scores for each data point
    - Reliability ratings based on data point characteristics
    - Confidence scores for individual measurements
    - Performance ratings for actions/results
    - Historical rating trends for data points
    """
    
    def __init__(self):
        # Data point ratings and scores
        self.data_point_ratings: Dict[str, float] = {}  # ID -> rating score
        self.data_point_confidence: Dict[str, float] = {}  # ID -> confidence score
        self.data_point_reliability: Dict[str, float] = {}  # ID -> reliability score
        self.data_point_metadata: Dict[str, Dict[str, Any]] = {}  # ID -> metadata
        
        # Historical tracking
        self.rating_history: Dict[str, List[Tuple[str, float]]] = defaultdict(list)  # ID -> [(timestamp, rating)]
        self.performance_trends: Dict[str, float] = {}  # ID -> trend score
        
        # Rating categories
        self.rating_categories = {
            'action_quality': 'Quality of agent actions (color mixing precision)',
            'result_reliability': 'Reliability of color mixing results', 
            'measurement_accuracy': 'Accuracy of distance/gain/uncertainty measurements',
            'strategy_effectiveness': 'Effectiveness of policy decisions',
            'historical_consistency': 'Consistency with historical patterns'
        }
    
    def _initialize_baseline_quality_metrics(self):
        """Initialize baseline quality metrics for different data categories"""
        from datetime import datetime
        
        baseline_timestamp = datetime.now().isoformat()
        
        for category in self._data_quality_categories:
            metric_id = f"baseline_{category}_quality"
            self.quality_metrics[metric_id] = DataQualityMetric(
                metric_id=metric_id,
                description=f"Baseline quality assessment for {category} data",
                quality_score=0.5,  # Neutral starting point
                data_source=f"{category}_loader", 
                completeness=0.0,  # No data loaded yet
                validation_status='unvalidated',
                confidence_level=0.5,
                information_gain_quality=0.5,
                uncertainty_reliability=0.5,
                timestamp=baseline_timestamp
            )
            
            # Initialize tracking
            self.completeness_scores[category] = 0.0
            self.reliability_assessments[category] = 0.5
    
    def rate_data_point(self, data_point_id: str, data_point: Dict[str, Any], 
                        category: str = 'general') -> Dict[str, float]:
        """
        Rate and score an individual data point based on its characteristics and quality.
        
        Args:
            data_point_id: Unique identifier for this data point
            data_point: The data point to be rated
            category: Category of data point (action_quality, result_reliability, etc.)
            
        Returns:
            Dict containing various scores for the data point
        """
        from datetime import datetime
        
        scores = {
            'overall_rating': 0.5,  # Overall quality rating (0-1)
            'confidence_score': 0.5,  # Confidence in this data point
            'reliability_score': 0.5,  # How reliable/consistent this data point is
            'performance_score': 0.5,  # Performance quality of this data point
            'trend_score': 0.5  # How this data point fits with trends
        }
        
        # Rate based on category
        if category == 'action_quality':
            scores = self._rate_action_quality(data_point)
        elif category == 'result_reliability':  
            scores = self._rate_result_reliability(data_point)
        elif category == 'measurement_accuracy':
            scores = self._rate_measurement_accuracy(data_point)
        elif category == 'strategy_effectiveness':
            scores = self._rate_strategy_effectiveness(data_point)
        elif category == 'historical_consistency':
            scores = self._rate_historical_consistency(data_point_id, data_point)
        else:
            scores = self._rate_general_data_point(data_point)
        
        # Store ratings
        self.data_point_ratings[data_point_id] = scores['overall_rating']
        self.data_point_confidence[data_point_id] = scores['confidence_score']
        self.data_point_reliability[data_point_id] = scores['reliability_score']
        
        # Store metadata
        self.data_point_metadata[data_point_id] = {
            'category': category,
            'timestamp': datetime.now().isoformat(),
            'scores': scores.copy()
        }
        
        # Update historical tracking
        timestamp = datetime.now().isoformat()
        self.rating_history[data_point_id].append((timestamp, scores['overall_rating']))
        
        # Update performance trends
        self._update_performance_trend(data_point_id)
        
        return scores
    
    def _assess_completeness(self, data_category: str, data_payload: Any) -> float:
        """Assess how complete the data is based on expected fields and structure"""
        if data_payload is None:
            return 0.0
        
        if data_category == 'system_analytics':
            if isinstance(data_payload, list) and data_payload:
                # Check for required fields in system analytics
                required_fields = ['agent_id', 'policy_name', 'agent_trueskill_mu', 
                                 'agent_trueskill_sigma', 'uncertainty', 'information_gain']
                sample_entry = data_payload[0] if data_payload else {}
                present_fields = sum(1 for field in required_fields if field in sample_entry)
                field_completeness = present_fields / len(required_fields)
                
                # Check data volume completeness
                volume_score = min(1.0, len(data_payload) / 50.0)  # Expect at least 50 entries for full score
                
                return (field_completeness + volume_score) / 2.0
            return 0.2
        
        elif data_category == 'match_data':
            if isinstance(data_payload, list) and data_payload:
                # Check for match data structure
                has_matches = any('matches' in entry for entry in data_payload if isinstance(entry, dict))
                volume_score = min(1.0, len(data_payload) / 10.0)  # Expect at least 10 match entries
                
                return (0.8 if has_matches else 0.2) + volume_score * 0.2
            return 0.1
        
        elif data_category == 'game_results':
            if isinstance(data_payload, list) and data_payload:
                # Check for game results structure
                sample_entry = data_payload[0] if data_payload else {}
                has_actions = 'actions' in sample_entry
                has_results = 'results' in sample_entry
                has_ratings = 'ratings_before' in sample_entry or 'ratings_after' in sample_entry
                
                structure_score = (has_actions + has_results + has_ratings) / 3.0
                volume_score = min(1.0, len(data_payload) / 20.0)  # Expect at least 20 game results
                
                return (structure_score + volume_score) / 2.0
            return 0.1
        
        # Generic assessment for other categories
        if isinstance(data_payload, (list, dict)) and data_payload:
            return min(1.0, len(data_payload) / 10.0)  # Generic volume assessment
        
        return 0.0
    
    def _assess_information_metrics(self, data_payload: Any) -> Tuple[float, float]:
        """Assess the quality of information gain and uncertainty metrics in the data"""
        info_gain_quality = 0.5  # Default neutral
        uncertainty_reliability = 0.5  # Default neutral
        
        if isinstance(data_payload, list) and data_payload:
            info_gains = []
            uncertainties = []
            
            for entry in data_payload:
                if isinstance(entry, dict):
                    # Extract information gain values
                    if 'information_gain' in entry:
                        info_gains.append(entry['information_gain'])
                    # Extract uncertainty values
                    if 'uncertainty' in entry:
                        uncertainties.append(entry['uncertainty'])
                    
                    # Check nested results/actions for more metrics
                    if 'results' in entry:
                        for result in entry['results'].values() if isinstance(entry['results'], dict) else []:
                            if isinstance(result, dict):
                                if 'information_gain' in result:
                                    info_gains.append(result['information_gain'])
                                if 'uncertainty' in result:
                                    uncertainties.append(result['uncertainty'])
            
            # Assess information gain quality
            if info_gains:
                info_gains = [ig for ig in info_gains if isinstance(ig, (int, float)) and not np.isnan(ig)]
                if info_gains:
                    # Good quality if values are reasonable and varied
                    mean_ig = np.mean(info_gains)
                    std_ig = np.std(info_gains)
                    
                    # Quality based on reasonable range and variation
                    if 0 <= mean_ig <= 10000 and std_ig > 0.1:  # Reasonable range with variation
                        info_gain_quality = 0.9
                    elif 0 <= mean_ig <= 50000:  # Reasonable but possibly high
                        info_gain_quality = 0.7
                    else:  # Extreme values may indicate issues
                        info_gain_quality = 0.3
            
            # Assess uncertainty reliability
            if uncertainties:
                uncertainties = [u for u in uncertainties if isinstance(u, (int, float)) and not np.isnan(u)]
                if uncertainties:
                    mean_unc = np.mean(uncertainties)
                    std_unc = np.std(uncertainties)
                    
                    # Quality based on TrueSkill-appropriate uncertainty values
                    if 10 <= mean_unc <= 500 and std_unc > 1.0:  # Typical TrueSkill uncertainty range
                        uncertainty_reliability = 0.9
                    elif 0 <= mean_unc <= 1000:  # Reasonable extended range
                        uncertainty_reliability = 0.7
                    else:  # Values outside expected range
                        uncertainty_reliability = 0.3
        
        return info_gain_quality, uncertainty_reliability
    
    def _log_data_loading(self, data_category: str, quality_metric: DataQualityMetric, 
                          loading_success: bool, validation_errors: List[str]):
        """Log data loading events for tracking and analysis"""
        log_entry = {
            'timestamp': quality_metric.timestamp,
            'category': data_category,
            'loading_success': loading_success,
            'quality_score': quality_metric.quality_score,
            'completeness': quality_metric.completeness,
            'validation_status': quality_metric.validation_status,
            'validation_errors': validation_errors,
            'info_gain_quality': quality_metric.information_gain_quality,
            'uncertainty_reliability': quality_metric.uncertainty_reliability
        }
        
        self.data_loading_log.append(log_entry)
        
    def _rate_action_quality(self, data_point: Dict[str, Any]) -> Dict[str, float]:
        """Rate the quality of an agent action data point"""
        scores = {'overall_rating': 0.5, 'confidence_score': 0.5, 'reliability_score': 0.5, 'performance_score': 0.5, 'trend_score': 0.5}
        
        if 'action' in data_point:
            action = data_point['action']
            if isinstance(action, dict):
                # Check action validity
                has_required_fields = all(field in action for field in ['red_percent', 'yellow_percent', 'blue_percent', 'water_percent'])
                
                if has_required_fields:
                    # Rate based on action characteristics
                    total_percent = sum([action.get('red_percent', 0), action.get('yellow_percent', 0), 
                                       action.get('blue_percent', 0), action.get('water_percent', 0)])
                    
                    # Good actions should sum close to 100%
                    if 95 <= total_percent <= 105:
                        scores['reliability_score'] = 0.9
                        scores['overall_rating'] = 0.8
                    elif 80 <= total_percent <= 120:
                        scores['reliability_score'] = 0.7
                        scores['overall_rating'] = 0.6
                    else:
                        scores['reliability_score'] = 0.3
                        scores['overall_rating'] = 0.3
                    
                    # Rate performance based on color distance if available
                    if 'color_distance' in data_point:
                        distance = data_point['color_distance']
                        if distance < 0.1:  # Very precise
                            scores['performance_score'] = 0.95
                        elif distance < 0.3:  # Good
                            scores['performance_score'] = 0.8
                        elif distance < 0.6:  # Average
                            scores['performance_score'] = 0.5
                        else:  # Poor
                            scores['performance_score'] = 0.2
                        
                        scores['overall_rating'] = (scores['reliability_score'] + scores['performance_score']) / 2.0
        
        return scores
    
    def _rate_result_reliability(self, data_point: Dict[str, Any]) -> Dict[str, float]:
        """Rate the reliability of a color mixing result"""
        scores = {'overall_rating': 0.5, 'confidence_score': 0.5, 'reliability_score': 0.5, 'performance_score': 0.5, 'trend_score': 0.5}
        
        if 'actual_rgb' in data_point and 'target_rgb' in data_point:
            actual = data_point['actual_rgb']
            target = data_point['target_rgb']
            
            if len(actual) == 3 and len(target) == 3:
                # Check if RGB values are in valid range
                valid_actual = all(0 <= val <= 1.2 for val in actual)  # Allow slight overflow
                valid_target = all(0 <= val <= 1.2 for val in target)
                
                if valid_actual and valid_target:
                    scores['reliability_score'] = 0.9
                    
                    # Rate based on color distance
                    color_distance = sum(abs(a - t) for a, t in zip(actual, target)) / 3.0
                    if color_distance < 0.05:
                        scores['performance_score'] = 0.95
                    elif color_distance < 0.15:
                        scores['performance_score'] = 0.8
                    elif color_distance < 0.4:
                        scores['performance_score'] = 0.6
                    else:
                        scores['performance_score'] = 0.3
                else:
                    scores['reliability_score'] = 0.2
                    scores['performance_score'] = 0.2
        
        scores['overall_rating'] = (scores['reliability_score'] + scores['performance_score']) / 2.0
        return scores
    
    def _rate_measurement_accuracy(self, data_point: Dict[str, Any]) -> Dict[str, float]:
        """Rate the accuracy of measurements (information_gain, uncertainty, etc.)"""
        scores = {'overall_rating': 0.5, 'confidence_score': 0.5, 'reliability_score': 0.5, 'performance_score': 0.5, 'trend_score': 0.5}
        
        # Check information_gain quality
        if 'information_gain' in data_point:
            info_gain = data_point['information_gain']
            if isinstance(info_gain, (int, float)) and not np.isnan(info_gain):
                if 0 <= info_gain <= 10000:  # Reasonable range
                    scores['reliability_score'] = 0.9
                elif 0 <= info_gain <= 50000:  # Extended range
                    scores['reliability_score'] = 0.7
                else:
                    scores['reliability_score'] = 0.3
        
        # Check uncertainty quality  
        if 'uncertainty' in data_point:
            uncertainty = data_point['uncertainty']
            if isinstance(uncertainty, (int, float)) and not np.isnan(uncertainty):
                if 10 <= uncertainty <= 500:  # TrueSkill typical range
                    scores['confidence_score'] = 0.9
                elif 0 <= uncertainty <= 1000:  # Extended range
                    scores['confidence_score'] = 0.7
                else:
                    scores['confidence_score'] = 0.3
        
        scores['overall_rating'] = (scores['reliability_score'] + scores['confidence_score']) / 2.0
        return scores
    
    def _rate_strategy_effectiveness(self, data_point: Dict[str, Any]) -> Dict[str, float]:
        """Rate the effectiveness of a policy/strategy decision"""
        scores = {'overall_rating': 0.5, 'confidence_score': 0.5, 'reliability_score': 0.5, 'performance_score': 0.5, 'trend_score': 0.5}
        
        if 'policy_used' in data_point:
            # Rate based on reward/performance
            if 'reward' in data_point:
                reward = data_point['reward']
                if reward > 0.8:  # High reward
                    scores['performance_score'] = 0.9
                elif reward > 0.5:  # Medium reward
                    scores['performance_score'] = 0.7
                elif reward > 0.2:  # Low reward
                    scores['performance_score'] = 0.4
                else:  # Very low reward
                    scores['performance_score'] = 0.2
            
            # Rate based on efficiency if available
            if 'efficiency_score' in data_point:
                efficiency = data_point['efficiency_score']
                if efficiency > 0.1:  # Good efficiency
                    scores['reliability_score'] = 0.8
                else:  # Poor efficiency
                    scores['reliability_score'] = 0.4
        
        scores['overall_rating'] = (scores['performance_score'] + scores['reliability_score']) / 2.0
        return scores
    
    def _rate_historical_consistency(self, data_point_id: str, data_point: Dict[str, Any]) -> Dict[str, float]:
        """Rate how consistent this data point is with historical patterns"""
        scores = {'overall_rating': 0.5, 'confidence_score': 0.5, 'reliability_score': 0.5, 'performance_score': 0.5, 'trend_score': 0.5}
        
        # Check against historical data for this ID
        if data_point_id in self.rating_history:
            history = self.rating_history[data_point_id]
            if len(history) >= 3:
                # Calculate consistency with previous ratings
                recent_ratings = [rating for _, rating in history[-5:]]  # Last 5 ratings
                avg_rating = np.mean(recent_ratings)
                std_rating = np.std(recent_ratings)
                
                # Current overall score estimate
                current_estimate = 0.5  # Will be updated by other rating methods
                
                # High consistency if new estimate is within 2 standard deviations
                if std_rating > 0:
                    deviation = abs(current_estimate - avg_rating) / std_rating
                    if deviation < 1.0:  # Within 1 std dev
                        scores['trend_score'] = 0.9
                    elif deviation < 2.0:  # Within 2 std dev
                        scores['trend_score'] = 0.7
                    else:  # Outside 2 std dev
                        scores['trend_score'] = 0.3
        
        scores['overall_rating'] = scores['trend_score']
        return scores
    
    def _rate_general_data_point(self, data_point: Dict[str, Any]) -> Dict[str, float]:
        """General rating for unspecified data point types"""
        scores = {'overall_rating': 0.5, 'confidence_score': 0.5, 'reliability_score': 0.5, 'performance_score': 0.5, 'trend_score': 0.5}
        
        # Basic completeness check
        if isinstance(data_point, dict) and data_point:
            field_count = len(data_point)
            if field_count >= 5:  # Rich data point
                scores['reliability_score'] = 0.8
            elif field_count >= 3:  # Adequate data point
                scores['reliability_score'] = 0.6
            else:  # Sparse data point
                scores['reliability_score'] = 0.3
            
            scores['overall_rating'] = scores['reliability_score']
        
        return scores
    
    def _update_performance_trend(self, data_point_id: str):
        """Update performance trend for a data point based on its rating history"""
        if data_point_id in self.rating_history:
            history = self.rating_history[data_point_id]
            if len(history) >= 3:
                # Calculate trend from recent ratings
                recent_ratings = [rating for _, rating in history[-5:]]
                if len(recent_ratings) >= 3:
                    # Simple trend calculation: improvement, stable, or declining
                    first_half = np.mean(recent_ratings[:len(recent_ratings)//2])
                    second_half = np.mean(recent_ratings[len(recent_ratings)//2:])
                    
                    trend_diff = second_half - first_half
                    if trend_diff > 0.1:
                        self.performance_trends[data_point_id] = 0.8  # Improving
                    elif trend_diff < -0.1:
                        self.performance_trends[data_point_id] = 0.3  # Declining  
                    else:
                        self.performance_trends[data_point_id] = 0.6  # Stable
    
    def evaluate_collective_contribution(self, data_points: List[Dict[str, Any]], 
                                       memory_limit: int = 200) -> Dict[str, Any]:
        """
        Evaluate how data points collectively contribute to TrueSkill and Blade-Chest systems
        given limited inventory/memory constraints.
        
        Args:
            data_points: List of data points to evaluate
            memory_limit: Maximum number of data points to keep in memory
            
        Returns:
            Dict with collective contribution analysis
        """
        contribution_analysis = {
            'total_data_points': len(data_points),
            'memory_utilization': min(1.0, len(data_points) / memory_limit),
            'trueskill_contribution_score': 0.0,
            'blade_chest_contribution_score': 0.0,
            'information_quality_score': 0.0,
            'memory_efficiency_score': 0.0,
            'recommended_retention': [],  # Data points to keep
            'recommended_pruning': []     # Data points to remove
        }
        
        # Rate each data point and calculate contributions
        data_point_contributions = []
        
        for i, data_point in enumerate(data_points):
            point_id = f"data_point_{i}"
            
            # Rate the data point
            ratings = self.rate_data_point(point_id, data_point, 'measurement_accuracy')
            
            # Calculate contribution to collective systems
            trueskill_contribution = self._calculate_trueskill_contribution(data_point, ratings)
            blade_chest_contribution = self._calculate_blade_chest_contribution(data_point, ratings)
            
            contribution_record = {
                'index': i,
                'data_point': data_point,
                'ratings': ratings,
                'trueskill_contribution': trueskill_contribution,
                'blade_chest_contribution': blade_chest_contribution,
                'collective_value': (trueskill_contribution + blade_chest_contribution) / 2.0
            }
            
            data_point_contributions.append(contribution_record)
        
        # Calculate overall scores
        if data_point_contributions:
            contribution_analysis['trueskill_contribution_score'] = np.mean([
                dp['trueskill_contribution'] for dp in data_point_contributions
            ])
            contribution_analysis['blade_chest_contribution_score'] = np.mean([
                dp['blade_chest_contribution'] for dp in data_point_contributions  
            ])
            contribution_analysis['information_quality_score'] = np.mean([
                dp['ratings']['overall_rating'] for dp in data_point_contributions
            ])
        
        # Memory efficiency scoring
        if len(data_points) <= memory_limit:
            contribution_analysis['memory_efficiency_score'] = 1.0  # All fit in memory
        else:
            # Need to prioritize - rank by collective value
            sorted_contributions = sorted(
                data_point_contributions, 
                key=lambda x: x['collective_value'], 
                reverse=True
            )
            
            # Keep top data points that fit in memory
            to_retain = sorted_contributions[:memory_limit]
            to_prune = sorted_contributions[memory_limit:]
            
            contribution_analysis['recommended_retention'] = [
                dp['index'] for dp in to_retain
            ]
            contribution_analysis['recommended_pruning'] = [
                dp['index'] for dp in to_prune
            ]
            
            # Efficiency score based on retained quality vs total quality
            retained_quality = np.mean([dp['collective_value'] for dp in to_retain])
            total_quality = np.mean([dp['collective_value'] for dp in data_point_contributions])
            contribution_analysis['memory_efficiency_score'] = retained_quality / total_quality if total_quality > 0 else 0.5
        
        return contribution_analysis
    
    def _calculate_trueskill_contribution(self, data_point: Dict[str, Any], ratings: Dict[str, float]) -> float:
        """Calculate how much this data point contributes to TrueSkill rating accuracy"""
        contribution = 0.5  # Base contribution
        
        # High-quality uncertainty and information gain measurements contribute more
        if 'uncertainty' in data_point and 'information_gain' in data_point:
            uncertainty = data_point.get('uncertainty', 0)
            info_gain = data_point.get('information_gain', 0)
            
            # TrueSkill benefits from reliable uncertainty measurements
            if isinstance(uncertainty, (int, float)) and 50 <= uncertainty <= 300:
                contribution += 0.2
            
            # TrueSkill benefits from meaningful information gain
            if isinstance(info_gain, (int, float)) and 100 <= info_gain <= 5000:
                contribution += 0.2
            
            # Bonus for high measurement accuracy rating
            contribution += ratings['reliability_score'] * 0.3
        
        return min(1.0, contribution)
    
    def _calculate_blade_chest_contribution(self, data_point: Dict[str, Any], ratings: Dict[str, float]) -> float:
        """Calculate how much this data point contributes to Blade-Chest intransitivity detection"""
        contribution = 0.5  # Base contribution
        
        # Blade-Chest benefits from reliable performance comparisons
        if 'reward' in data_point or 'color_distance' in data_point:
            # Clear performance signals help with pairwise comparisons
            if 'color_distance' in data_point:
                distance = data_point.get('color_distance', 1.0)
                if distance < 0.5:  # Clear performance signal
                    contribution += 0.25
            
            if 'reward' in data_point:
                reward = data_point.get('reward', 0)
                if reward > 0.3:  # Meaningful reward signal
                    contribution += 0.25
        
        # Agent-policy information helps with intransitivity patterns
        if 'agent_id' in data_point and 'policy_used' in data_point:
            contribution += 0.15  # Helps build comparison matrix
        
        # Bonus for high performance rating
        contribution += ratings['performance_score'] * 0.2
        
        return min(1.0, contribution)
    
    def get_data_point_rating(self, data_point_id: str) -> Optional[float]:
        """Get the overall rating for a specific data point"""
        return self.data_point_ratings.get(data_point_id)
    
    def get_data_point_scores(self, data_point_id: str) -> Optional[Dict[str, float]]:
        """Get all scores for a specific data point"""
        if data_point_id in self.data_point_metadata:
            return self.data_point_metadata[data_point_id].get('scores')
        return None
    
    def get_top_rated_data_points(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get the top-rated data points"""
        sorted_ratings = sorted(self.data_point_ratings.items(), key=lambda x: x[1], reverse=True)
        return sorted_ratings[:limit]
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get a summary of the data point rating system"""
        return {
            'total_data_points_rated': len(self.data_point_ratings),
            'average_rating': np.mean(list(self.data_point_ratings.values())) if self.data_point_ratings else 0.0,
            'rating_distribution': {
                'high_quality': len([r for r in self.data_point_ratings.values() if r > 0.7]),
                'medium_quality': len([r for r in self.data_point_ratings.values() if 0.3 <= r <= 0.7]),
                'low_quality': len([r for r in self.data_point_ratings.values() if r < 0.3])
            },
            'categories_used': list(self.rating_categories.keys()),
            'performance_trends': len(self.performance_trends)
        }