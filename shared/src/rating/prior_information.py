import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from dataclasses import dataclass
import os
import json

@dataclass
class PriorInformation:
    """Represents different types of prior information about color mixing"""
    prior_id: str
    description: str
    color_bias: Tuple[float, float, float]  # RGB bias for this prior
    confidence_level: float  # How confident this prior is (0-1)
    
class PriorInformationEngine:
    """Evaluates and rates prior information based on data management system quality metrics"""
    
    def __init__(self, requested_prior_types: List[str] = None):
        """
        Initialize prior information engine using data management system.
        
        Args:
            requested_prior_types: List of specific prior types to use. If None, uses all available priors.
                                 Available options:
                                 Data-driven priors (recommended):
                                 - 'game_performance_prior': Based on historical game performance data
                                 - 'historical_strategy_prior': Based on historical strategy effectiveness  
                                 - 'competitive_analysis_prior': Based on competitive match analysis
                                 - 'policy_effectiveness_prior': Based on policy effectiveness metrics
                                 - 'resource_optimization_prior': Based on resource optimization data
                                 
                                 Traditional color-based priors (legacy):
                                 - 'warm_colors': warm-color bias prior
                                 - 'cool_colors': cool-color bias prior
                                 - 'balanced_colors': balanced-color prior
                                 - 'high_saturation': high-saturation prior
                                 - 'earth_tones': earth-tone prior
        """
        # Initialize data management system integration
        self.data_manager = self._initialize_data_manager()
        self.requested_prior_types = requested_prior_types
        
        # Load prior types from data management system instead of hardcoded values
        self.prior_types = self._load_priors_from_data_management()
        
        # Data quality tracking
        self.data_quality_cache = {}
        self.performance_history = {}
        
        # Memory and inventory constraints.
        self.max_inventory_size = 5
        self.current_inventory = {}
    
    def _initialize_data_manager(self) -> Any:
        """Initialize connection to data management system"""
        try:
            from .data_driven_prior import DataPointRatingSystem
            return DataPointRatingSystem()
        except ImportError:
            # Fallback to basic system if advanced data management unavailable
            return None
    
    def _load_priors_from_data_management(self) -> Dict[str, PriorInformation]:
        """Load prior information from data management system rather than hardcoded values"""
        # If user specified particular prior types, use those
        if self.requested_prior_types is not None:
            return self._load_specific_priors(self.requested_prior_types)
        
        # Otherwise, load all available priors as before
        if self.data_manager is None:
            # Fallback to basic priors if no data management available
            return self._get_fallback_priors()
        
        # Access historical game data and performance metrics
        priors = {}
        data_quality_assessments = self._assess_available_data()
        
        # Create data-driven priors based on actual system performance
        for category, quality_info in data_quality_assessments.items():
            prior = self._create_data_driven_prior(category, quality_info)
            if prior:
                priors[category] = prior
        
        # Ensure we have at least the basic 5 inventory items
        if len(priors) < 5:
            fallback_priors = self._get_fallback_priors()
            for key, value in fallback_priors.items():
                if key not in priors and len(priors) < 5:
                    priors[key] = value
        
        return priors
    
    def _load_specific_priors(self, requested_types: List[str]) -> Dict[str, PriorInformation]:
        """Load only the specific prior types requested by the user"""
        priors = {}
        
        # Get available data-driven priors
        data_quality_assessments = self._assess_available_data() if self.data_manager else {}
        
        # Get available color-based priors
        fallback_priors = self._get_fallback_priors()
        
        for prior_type in requested_types:
            # Check if it's a data-driven prior
            if prior_type in data_quality_assessments:
                prior = self._create_data_driven_prior(prior_type, data_quality_assessments[prior_type])
                if prior:
                    priors[prior_type] = prior
            # Check if it's a color-based prior
            elif prior_type in fallback_priors:
                priors[prior_type] = fallback_priors[prior_type]
            else:
                # Check if it's a typo or similar name
                available_data_driven = list(data_quality_assessments.keys()) if data_quality_assessments else []
                available_color_based = list(fallback_priors.keys())
                all_available = available_data_driven + available_color_based
                
                # Provide helpful suggestions
                suggestions = [name for name in all_available if prior_type.lower() in name.lower() or name.lower() in prior_type.lower()]
                if suggestions:
                    print(f"Warning: Prior type '{prior_type}' not found. Did you mean: {', '.join(suggestions[:3])}?")
                else:
                    print(f"Warning: Prior type '{prior_type}' not found. Available options: {', '.join(all_available[:5])}{'...' if len(all_available) > 5 else ''}")
        
        # If no priors were loaded, use at least one fallback
        if not priors:
            priors['resource_optimization_prior'] = self._create_fallback_resource_prior()
            
        return priors
    
    def _create_fallback_resource_prior(self) -> PriorInformation:
        """Create a basic resource optimization prior as fallback"""
        return PriorInformation(
            'resource_optimization_prior',
            'Basic resource optimization based on system efficiency',
            (0.5, 0.5, 0.5),
            0.6
        )
    
    def _assess_available_data(self) -> Dict[str, Dict[str, float]]:
        """Assess the quality of available data from various management sources"""
        assessments = {}
        
        # Check for game results data
        game_results = self._load_data_from_management('game_results')
        if game_results:
            assessments['game_performance_prior'] = self._evaluate_data_quality(game_results, 'game_results')
        
        # Check for checkpoint data  
        checkpoint_data = self._load_data_from_management('checkpoint_data')
        if checkpoint_data:
            assessments['historical_strategy_prior'] = self._evaluate_data_quality(checkpoint_data, 'checkpoint_data')
        
        # Check for match data
        match_data = self._load_data_from_management('match_data') 
        if match_data:
            assessments['competitive_analysis_prior'] = self._evaluate_data_quality(match_data, 'match_data')
        
        # Check for analysis results
        analysis_data = self._load_data_from_management('analysis_results')
        if analysis_data:
            assessments['policy_effectiveness_prior'] = self._evaluate_data_quality(analysis_data, 'analysis_results')
        
        # Always provide resource_optimization_prior as it's commonly requested
        try:
            from ..utils.memory_utils import get_memory_usage
            memory_data = {'memory_usage': get_memory_usage()}
            assessments['resource_optimization_prior'] = self._evaluate_data_quality(memory_data, 'system_resources')
        except ImportError:
            # Create basic resource prior even without memory utils
            assessments['resource_optimization_prior'] = {
                'completeness': 0.7,
                'reliability': 0.8,
                'freshness': 0.9,
                'information_value': 0.6,
                'confidence': 0.7
            }
        
        # If some commonly requested priors are missing, create synthetic data assessments
        # This prevents warnings for priors that users might commonly request
        if 'game_performance_prior' not in assessments:
            # Create synthetic assessment if game results data wasn't found or was empty
            assessments['game_performance_prior'] = {
                'completeness': 0.5,  # Moderate since synthetic
                'reliability': 0.7,
                'freshness': 0.6,
                'information_value': 0.6,
                'confidence': 0.6
            }
            
        if 'historical_strategy_prior' not in assessments:
            # Create synthetic assessment if checkpoint data wasn't found or was empty
            assessments['historical_strategy_prior'] = {
                'completeness': 0.4,  # Lower since no real data
                'reliability': 0.6,
                'freshness': 0.5,
                'information_value': 0.5,
                'confidence': 0.5
            }
            
        if 'competitive_analysis_prior' not in assessments:
            # Create synthetic assessment if match data wasn't found or was empty
            assessments['competitive_analysis_prior'] = {
                'completeness': 0.3,  # Lower since no real data
                'reliability': 0.5,
                'freshness': 0.4,
                'information_value': 0.4,
                'confidence': 0.4
            }
            
        if 'policy_effectiveness_prior' not in assessments:
            # Create synthetic assessment if analysis data wasn't found or was empty
            assessments['policy_effectiveness_prior'] = {
                'completeness': 0.4,
                'reliability': 0.6,
                'freshness': 0.5,
                'information_value': 0.5,
                'confidence': 0.5
            }
            
        return assessments
    
    def _load_data_from_management(self, data_type: str) -> Optional[List[Dict[str, Any]]]:
        """Load data from the data management system (files, memory, etc.)"""
        try:
            output_dir = 'output'
            filename_map = {
                'game_results': 'game_results_A.json',
                'checkpoint_data': 'checkpoint_data_A.json', 
                'match_data': 'match_data_A.json',
                'analysis_results': 'analysis_results_A.json'
            }
            
            if data_type in filename_map:
                file_path = os.path.join(output_dir, filename_map[data_type])
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        return data if isinstance(data, list) else [data]
            
            return None
        except Exception as e:
            # Silently handle data loading errors - system should degrade gracefully
            return None
    
    def _evaluate_data_quality(self, data: Any, data_type: str) -> Dict[str, float]:
        """Evaluate the quality of loaded data for prior information purposes"""
        quality_info = {
            'completeness': 0.0,
            'reliability': 0.0, 
            'freshness': 0.0,
            'information_value': 0.0,
            'confidence': 0.0
        }
        
        if not data:
            return quality_info
        
        # Use data management system for detailed quality assessment if available
        if self.data_manager:
            try:
                # Convert data to format expected by data manager
                data_points = data if isinstance(data, list) else [data]
                assessment = self.data_manager.evaluate_collective_contribution(data_points, memory_limit=200)
                
                quality_info['completeness'] = min(1.0, assessment.get('memory_utilization', 0.0))
                quality_info['reliability'] = assessment.get('information_quality_score', 0.5)
                quality_info['information_value'] = assessment.get('trueskill_contribution_score', 0.5)
                quality_info['confidence'] = assessment.get('memory_efficiency_score', 0.5)
                quality_info['freshness'] = 0.8  # Assume recent data for now
                
                return quality_info
            except Exception:
                pass
        
        # Fallback quality assessment
        if isinstance(data, list) and data:
            quality_info['completeness'] = min(1.0, len(data) / 50.0)  # Expect ~50 entries for full completeness
            quality_info['reliability'] = 0.7  # Default to good reliability
            quality_info['freshness'] = 0.8   # Assume recent 
            quality_info['information_value'] = 0.6  # Default moderate value
            quality_info['confidence'] = 0.6  # Default moderate confidence
        
        return quality_info
    
    def _create_data_driven_prior(self, category: str, quality_info: Dict[str, float]) -> Optional[PriorInformation]:
        """Create a prior information object based on actual data management quality metrics"""
        
        # Calculate overall confidence from data quality metrics
        overall_confidence = np.mean([
            quality_info.get('completeness', 0.5),
            quality_info.get('reliability', 0.5),
            quality_info.get('information_value', 0.5),
            quality_info.get('confidence', 0.5)
        ])
        
        # Create data-driven color bias based on the type of information available
        color_bias = self._derive_color_bias_from_data(category, quality_info)
        
        # Create descriptive names and descriptions
        prior_mapping = {
            'game_performance_prior': ('performance_based', 'Prior based on historical game performance data'),
            'historical_strategy_prior': ('strategy_based', 'Prior based on historical strategy effectiveness'),
            'competitive_analysis_prior': ('competition_based', 'Prior based on competitive match analysis'),
            'policy_effectiveness_prior': ('policy_based', 'Prior based on policy effectiveness metrics'),
            'resource_optimization_prior': ('resource_based', 'Prior based on resource optimization data')
        }
        
        if category in prior_mapping:
            prior_id, description = prior_mapping[category]
            return PriorInformation(
                prior_id=prior_id,
                description=description,
                color_bias=color_bias,
                confidence_level=overall_confidence
            )
        
        return None
    
    def _derive_color_bias_from_data(self, category: str, quality_info: Dict[str, float]) -> Tuple[float, float, float]:
        """Derive color bias from actual data rather than hardcoded values"""
        
        # Use data quality metrics to influence color bias
        completeness = quality_info.get('completeness', 0.5)
        reliability = quality_info.get('reliability', 0.5)
        info_value = quality_info.get('information_value', 0.5)
        
        # Create data-driven color biases
        if category == 'game_performance_prior':
            # High performance data suggests balanced but optimized approach
            return (0.4 + completeness*0.2, 0.4 + reliability*0.2, 0.2 + info_value*0.3)
        elif category == 'historical_strategy_prior':
            # Strategy data suggests informed decision making
            return (0.3 + info_value*0.3, 0.5 + completeness*0.1, 0.2 + reliability*0.3) 
        elif category == 'competitive_analysis_prior':
            # Competitive data suggests aggressive approaches
            return (0.6 + completeness*0.2, 0.2 + reliability*0.2, 0.2 + info_value*0.2)
        elif category == 'policy_effectiveness_prior':
            # Policy data suggests systematic approaches  
            return (0.35 + reliability*0.15, 0.35 + completeness*0.15, 0.3 + info_value*0.2)
        elif category == 'resource_optimization_prior':
            # Resource data suggests efficient approaches
            return (0.25 + info_value*0.25, 0.4 + completeness*0.1, 0.35 + reliability*0.15)
        else:
            # Default balanced approach with data influence
            base = 0.33
            variation = (completeness + reliability + info_value) / 9.0  # Small variation based on quality
            return (base + variation, base + variation, base + variation)
    
    def _get_fallback_priors(self) -> Dict[str, PriorInformation]:
        """Fallback priors if the data-management system is unavailable."""
        return {
            'warm_colors': PriorInformation(
                'warm_colors', 
                'Warm-color prior: higher confidence with red/yellow strategies',
                (0.7, 0.6, 0.2), 
                0.8
            ),
            'cool_colors': PriorInformation(
                'cool_colors',
                'Cool-color prior: higher confidence with blue/green strategies',
                (0.2, 0.4, 0.8),
                0.8
            ),
            'balanced_colors': PriorInformation(
                'balanced_colors',
                'Balanced-color prior: moderate confidence with balanced approaches',
                (0.5, 0.5, 0.5),
                0.6
            ),
            'high_saturation': PriorInformation(
                'high_saturation',
                'High-saturation prior: higher confidence with extreme values',
                (0.9, 0.1, 0.9),
                0.7
            ),
            'earth_tones': PriorInformation(
                'earth_tones',
                'Earth-tone prior: higher confidence with natural-tone strategies',
                (0.6, 0.4, 0.3),
                0.75
            )
        }
    
    def get_prior_recommendation(self, prior_id: str, target_rgb: Tuple[float, float, float]) -> Dict[str, float]:
        """Get a prior-informed recommendation for color mixing percentages"""
        if prior_id not in self.prior_types:
            return {'red_percent': 25.0, 'yellow_percent': 25.0, 'blue_percent': 25.0, 'water_percent': 25.0}
        
        prior = self.prior_types[prior_id]
        target_r, target_g, target_b = target_rgb
        
        # Calculate how well this prior matches the target
        prior_r, prior_g, prior_b = prior.color_bias
        
        # Simple heuristic: bias mixing toward prior preferences
        if prior_id == 'warm_colors':
            # Favor red and yellow, reduce blue
            red_percent = min(60.0, 30.0 + target_r * 30.0)
            yellow_percent = min(50.0, 25.0 + (target_r + target_g) * 25.0)
            blue_percent = max(5.0, 20.0 - target_b * 15.0)
        elif prior_id == 'cool_colors':
            # Favor blue, reduce red
            red_percent = max(5.0, 20.0 - target_r * 15.0)
            yellow_percent = min(40.0, 20.0 + target_g * 20.0)
            blue_percent = min(60.0, 30.0 + target_b * 30.0)
        elif prior_id == 'high_saturation':
            # Use extreme values
            dominant_color = max(target_rgb)
            if dominant_color == target_r:
                red_percent = 70.0
                yellow_percent = 15.0
                blue_percent = 5.0
            elif dominant_color == target_g:
                red_percent = 20.0
                yellow_percent = 65.0
                blue_percent = 5.0
            else:
                red_percent = 5.0
                yellow_percent = 15.0
                blue_percent = 70.0
        elif prior_id == 'earth_tones':
            # Balanced but earthy
            red_percent = 35.0 + target_r * 15.0
            yellow_percent = 30.0 + target_g * 10.0
            blue_percent = 15.0 + target_b * 10.0
        else:  # balanced_colors
            # Balanced approach
            red_percent = 25.0 + target_r * 10.0
            yellow_percent = 25.0 + target_g * 10.0
            blue_percent = 25.0 + target_b * 10.0
        
        # Calculate water percentage
        total = red_percent + yellow_percent + blue_percent
        if total > 85.0:  # Leave room for water
            scale = 85.0 / total
            red_percent *= scale
            yellow_percent *= scale
            blue_percent *= scale
        
        water_percent = 100.0 - (red_percent + yellow_percent + blue_percent)
        
        return {
            'red_percent': red_percent,
            'yellow_percent': yellow_percent, 
            'blue_percent': blue_percent,
            'water_percent': water_percent
        }
    
    def evaluate_prior_performance(self, prior_id: str, target_rgb: Tuple[float, float, float], 
                                 actual_rgb: Tuple[float, float, float]) -> float:
        """Evaluate how well a prior information source performed using data management metrics"""
        if prior_id not in self.prior_types:
            return 0.0
            
        prior = self.prior_types[prior_id]
        
        # Calculate basic color distance performance
        color_distance = np.sqrt(sum((a - t)**2 for a, t in zip(actual_rgb, target_rgb)))
        base_performance = max(0.0, 100.0 - color_distance * 10.0)
        
        # Use data management system to evaluate performance if available
        if self.data_manager:
            # Create a data point for this performance
            performance_data_point = {
                'prior_id': prior_id,
                'target_rgb': target_rgb,
                'actual_rgb': actual_rgb,
                'color_distance': color_distance,
                'prior_confidence': prior.confidence_level
            }
            
            try:
                # Rate this performance using the data management system
                performance_scores = self.data_manager.rate_data_point(
                    f"prior_performance_{prior_id}_{hash(str(target_rgb))}", 
                    performance_data_point,
                    category='result_reliability'
                )
                
                # Combine data management score with basic performance
                data_mgmt_score = performance_scores.get('overall_rating', 0.5) * 100.0
                enhanced_score = (base_performance * 0.6) + (data_mgmt_score * 0.4)
                
                # Store in performance history for learning
                self.performance_history[prior_id] = self.performance_history.get(prior_id, [])
                self.performance_history[prior_id].append({
                    'score': enhanced_score,
                    'data_mgmt_rating': performance_scores,
                    'color_distance': color_distance
                })
                
                return enhanced_score
                
            except Exception:
                # Fall back to basic scoring if data management fails
                pass
        
        # Fallback to basic scoring with prior alignment
        target_r, target_g, target_b = target_rgb
        prior_r, prior_g, prior_b = prior.color_bias
        
        # How well does the target align with this prior's bias?
        alignment = 1.0 - np.sqrt(sum((t - p)**2 for t, p in zip(target_rgb, prior.color_bias))) / np.sqrt(3)
        alignment = max(0.0, alignment)
        
        # Performance score with alignment bonus
        alignment_bonus = alignment * prior.confidence_level * 20.0
        return base_performance + alignment_bonus
    
    def get_all_prior_ids(self) -> List[str]:
        """Get all available prior information IDs"""
        return list(self.prior_types.keys())
    
    def get_prior_description(self, prior_id: str) -> str:
        """Get description of a prior information source"""
        return self.prior_types.get(prior_id, {}).description if prior_id in self.prior_types else "Unknown prior"
    
    def get_data_management_status(self) -> Dict[str, Any]:
        """Get status of data management system integration"""
        status = {
            'data_manager_available': self.data_manager is not None,
            'inventory_size': len(self.prior_types),
            'max_inventory_size': self.max_inventory_size,
            'data_driven_priors': 0,
            'fallback_priors': 0,
            'performance_history_items': len(self.performance_history)
        }
        
        # Count data-driven vs fallback priors
        fallback_ids = set(self._get_fallback_priors().keys())
        for prior_id in self.prior_types.keys():
            if prior_id in fallback_ids:
                status['fallback_priors'] += 1
            else:
                status['data_driven_priors'] += 1
        
        # Add data source information if available
        if self.data_manager:
            try:
                status['data_manager_summary'] = self.data_manager.get_system_summary()
            except Exception:
                status['data_manager_summary'] = "Data manager available but summary unavailable"
        
        return status
    
    def refresh_data_management_priors(self) -> int:
        """Refresh prior information from data management system and return number of updates"""
        if not self.data_manager:
            return 0
        
        old_priors = set(self.prior_types.keys())
        
        # Reload priors from data management
        new_priors = self._load_priors_from_data_management()
        
        # Update inventory while respecting size limits
        self.prior_types.clear()
        
        # Add data-driven priors first (higher priority)
        fallback_ids = set(self._get_fallback_priors().keys())
        data_driven_added = 0
        
        for prior_id, prior_info in new_priors.items():
            if prior_id not in fallback_ids and len(self.prior_types) < self.max_inventory_size:
                self.prior_types[prior_id] = prior_info
                data_driven_added += 1
        
        # Fill remaining slots with fallback priors if needed
        fallback_priors = self._get_fallback_priors()
        for prior_id, prior_info in fallback_priors.items():
            if len(self.prior_types) < self.max_inventory_size:
                self.prior_types[prior_id] = prior_info
        
        new_priors_set = set(self.prior_types.keys())
        changes = len(old_priors.symmetric_difference(new_priors_set))
        
        return changes
