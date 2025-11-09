"""
Universal Multi-Metal Alloy Composition Optimizer
Supports various alloy types: Steel, Aluminum, Copper, Titanium, Nickel, etc.
"""
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib

logger = logging.getLogger(__name__)

class UniversalAlloyOptimizer:
    """
    Universal alloy composition optimizer that works with multiple metal types.
    Supports: Steel, Aluminum, Copper, Titanium, Nickel, and custom alloys.
    """
    
    # Define metal systems and their typical elements
    ALLOY_SYSTEMS = {
        'steel': {
            'base_elements': ['Fe'],
            'alloying_elements': ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Ni', 'Mo', 'Cu', 'Al', 'V', 'W', 'Co'],
            'addable_elements': ['Fe', 'C', 'Si', 'Mn', 'Cr', 'Ni', 'Mo', 'Cu', 'Al'],
            'temp_range': (1400, 1600),
            'density': 7.85
        },
        'aluminum': {
            'base_elements': ['Al'],
            'alloying_elements': ['Cu', 'Si', 'Mg', 'Zn', 'Fe', 'Mn', 'Cr', 'Ti', 'Li'],
            'addable_elements': ['Al', 'Cu', 'Si', 'Mg', 'Zn'],
            'temp_range': (1200, 1300),
            'density': 2.70
        },
        'copper': {
            'base_elements': ['Cu'],
            'alloying_elements': ['Zn', 'Sn', 'Pb', 'Ni', 'Al', 'Fe', 'Si', 'P', 'Be'],
            'addable_elements': ['Cu', 'Zn', 'Sn', 'Ni', 'Al'],
            'temp_range': (1000, 1200),
            'density': 8.96
        },
        'titanium': {
            'base_elements': ['Ti'],
            'alloying_elements': ['Al', 'V', 'Fe', 'Mo', 'Cr', 'Sn', 'Zr', 'Nb'],
            'addable_elements': ['Ti', 'Al', 'V', 'Fe'],
            'temp_range': (1600, 1800),
            'density': 4.51
        },
        'nickel': {
            'base_elements': ['Ni'],
            'alloying_elements': ['Cr', 'Fe', 'Mo', 'Co', 'Al', 'Ti', 'W', 'Nb'],
            'addable_elements': ['Ni', 'Cr', 'Fe', 'Mo'],
            'temp_range': (1400, 1600),
            'density': 8.90
        },
        'magnesium': {
            'base_elements': ['Mg'],
            'alloying_elements': ['Al', 'Zn', 'Mn', 'Si', 'Ca', 'Zr', 'Y', 'Nd'],
            'addable_elements': ['Mg', 'Al', 'Zn', 'Mn'],
            'temp_range': (600, 800),
            'density': 1.74
        }
    }
    
    def __init__(self, alloy_type: str = 'auto', config: Optional[Dict] = None):
        self.alloy_type = alloy_type.lower()
        self.config = config or {}
        self.models = {}
        self.feature_names = []
        self.is_trained = False
        self.alloy_system = None
        self.elements = []
        self.addable_elements = []
        
        # Initialize alloy system
        if self.alloy_type != 'auto':
            self._set_alloy_system(self.alloy_type)
        
        # Initialize sub-models
        self._initialize_models()
    
    def _set_alloy_system(self, alloy_type: str):
        """Set the alloy system based on type."""
        if alloy_type in self.ALLOY_SYSTEMS:
            self.alloy_system = self.ALLOY_SYSTEMS[alloy_type]
            self.elements = (self.alloy_system['base_elements'] + 
                           self.alloy_system['alloying_elements'])
            self.addable_elements = self.alloy_system['addable_elements']
            logger.info("Set alloy system: %s", alloy_type)
        else:
            raise ValueError(f"Unsupported alloy type: {alloy_type}")
    
    def auto_detect_alloy_type(self, composition: Dict[str, float]) -> str:
        """Automatically detect alloy type based on composition."""
        # Find the dominant element
        dominant_element = max(composition.items(), key=lambda x: x[1])[0]
        
        # Map dominant element to alloy type
        element_to_alloy = {
            'Fe': 'steel',
            'Al': 'aluminum', 
            'Cu': 'copper',
            'Ti': 'titanium',
            'Ni': 'nickel',
            'Mg': 'magnesium'
        }
        
        detected_type = element_to_alloy.get(dominant_element, 'steel')  # Default to steel
        logger.info("Auto-detected alloy type: %s (dominant: %s)", detected_type, dominant_element)
        
        return detected_type
    
    def _initialize_models(self):
        """Initialize all sub-models for different predictions."""
        self.models['additions'] = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=200, random_state=42)
        )
        self.models['process'] = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=150, random_state=42)
        )
        self.models['performance'] = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, random_state=42)
        )
        self.models['properties'] = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, random_state=42)
        )
    
    def extract_features(self, input_data: Dict) -> np.ndarray:
        """Extract features from input JSON data - works with any alloy type."""
        # Auto-detect alloy type if not set
        if self.alloy_type == 'auto' or self.alloy_system is None:
            detected_type = self.auto_detect_alloy_type(input_data['spectrometer'])
            self._set_alloy_system(detected_type)
        
        if self.alloy_system is None:
            raise ValueError("Could not determine alloy system")
        
        features = []
        
        # Current composition features (pad with zeros for missing elements)
        spectrometer = input_data['spectrometer']
        for element in self.elements:
            features.append(spectrometer.get(element, 0.0))
        
        # Target composition features
        target = input_data['target_composition']
        for element in self.elements:
            features.append(target.get(element, 0.0))
        
        # Composition deviations
        for element in self.elements:
            current = spectrometer.get(element, 0.0)
            target_val = target.get(element, 0.0)
            deviation = current - target_val
            features.append(deviation)
        
        # Process parameters (temperature adjusted for alloy type)
        furnace = input_data['furnace_temp']
        temp_range = self.alloy_system['temp_range']
        normalized_temps = [
            (furnace['zone1'] - temp_range[0]) / (temp_range[1] - temp_range[0]),
            (furnace['zone2'] - temp_range[0]) / (temp_range[1] - temp_range[0]),
            (furnace['zone3'] - temp_range[0]) / (temp_range[1] - temp_range[0])
        ]
        features.extend(normalized_temps)
        
        # Temperature gradients
        temp_gradient_12 = furnace['zone2'] - furnace['zone1']
        temp_gradient_23 = furnace['zone3'] - furnace['zone2']
        avg_temp = (furnace['zone1'] + furnace['zone2'] + furnace['zone3']) / 3
        features.extend([avg_temp, temp_gradient_12, temp_gradient_23])
        
        # Stirrer parameters
        stirrer = input_data['stirrer']
        features.extend([
            stirrer['rpm'],
            stirrer.get('torque', 50),
            stirrer['time_min']
        ])
        
        # Material properties
        features.append(input_data['load_cell']['batch_weight_kg'])
        
        # Gas flow (if applicable)
        gas_flow = input_data.get('gas_flow', {})
        features.extend([
            gas_flow.get('O2_percent', 0.15),
            gas_flow.get('flow_L_per_min', 5.0)
        ])
        
        # Cooling parameters
        cooling = input_data.get('cooling', {})
        features.append(cooling.get('cool_rate_C_per_min', 15.0))
        
        # Previous dosing amounts (pad for missing elements)
        dosing = input_data.get('dosing', {})
        for element in self.addable_elements:
            features.append(dosing.get(f'{element}_added', 0.0))
        
        # Historical data
        if 'historical_data' in input_data:
            hist = input_data['historical_data']
            features.append(hist.get('previous_iterations', 0))
            features.append(hist.get('average_energy_consumption_kwh', 500))
            
            last_batch = hist.get('last_batch_additions', {})
            for element in self.addable_elements:
                features.append(last_batch.get(f'{element}_add_kg', 0.0))
        else:
            features.extend([0, 500] + [0] * len(self.addable_elements))
        
        # Alloy-specific features
        features.extend(self._calculate_alloy_specific_features(input_data))
        
        return np.array(features).reshape(1, -1)
    
    def _calculate_alloy_specific_features(self, input_data: Dict) -> List[float]:
        """Calculate alloy-specific derived features."""
        if self.alloy_system is None:
            return [0.0, 0.0, 0.0]  # Default features if no alloy system
            
        features = []
        current = input_data['spectrometer']
        
        if self.alloy_type == 'steel' or 'Fe' in current:
            # Steel-specific features
            features.extend([
                current.get('C', 0) / current.get('Fe', 1),  # C/Fe ratio
                current.get('Cr', 0) + current.get('Ni', 0),  # Austenite formers
                current.get('C', 0) * current.get('Cr', 0),   # Carbide potential
            ])
        elif self.alloy_type == 'aluminum' or 'Al' in current:
            # Aluminum-specific features
            features.extend([
                current.get('Cu', 0) / current.get('Al', 1),  # Cu/Al ratio
                current.get('Si', 0) + current.get('Mg', 0),  # Strengthening elements
                current.get('Zn', 0) / current.get('Al', 1),  # Zn/Al ratio
            ])
        elif self.alloy_type == 'copper' or 'Cu' in current:
            # Copper-specific features
            features.extend([
                current.get('Zn', 0) / current.get('Cu', 1),  # Zn/Cu ratio (brass)
                current.get('Sn', 0) / current.get('Cu', 1),  # Sn/Cu ratio (bronze)
                current.get('Ni', 0) + current.get('Al', 0),  # Strengthening elements
            ])
        else:
            # Generic features for other alloys
            base_element = self.alloy_system['base_elements'][0]
            base_content = current.get(base_element, 1.0)
            alloying_sum = sum(current.get(elem, 0) for elem in self.alloy_system['alloying_elements'])
            features.extend([
                alloying_sum / base_content,  # Total alloying ratio
                len([e for e in self.alloy_system['alloying_elements'] if current.get(e, 0) > 0.1]),  # Active elements
                0.0  # Placeholder
            ])
        
        return features
    
    def create_feature_names(self):
        """Create comprehensive feature names."""
        names = []
        
        # Current and target composition
        for prefix in ['current', 'target']:
            for element in self.elements:
                names.append(f'{prefix}_{element}')
        
        # Deviations
        for element in self.elements:
            names.append(f'deviation_{element}')
        
        # Process parameters
        names.extend(['norm_temp_zone1', 'norm_temp_zone2', 'norm_temp_zone3'])
        names.extend(['avg_temp', 'temp_gradient_12', 'temp_gradient_23'])
        names.extend(['stirrer_rpm', 'stirrer_torque', 'stirrer_time'])
        names.extend(['batch_weight', 'O2_percent', 'flow_rate', 'cooling_rate'])
        
        # Previous dosing
        for element in self.addable_elements:
            names.append(f'prev_{element}_added')
        
        # Historical
        names.extend(['prev_iterations', 'avg_energy'])
        for element in self.addable_elements:
            names.append(f'last_{element}_add')
        
        # Alloy-specific
        names.extend(['ratio_1', 'ratio_2', 'feature_3'])
        
        self.feature_names = names
        return names
    
    def generate_synthetic_training_data(self, n_samples: int = 5000) -> Tuple[np.ndarray, Dict]:
        """Generate synthetic training data for the current alloy system."""
        if not self.alloy_system:
            self._set_alloy_system('steel')  # Default
        
        if self.alloy_system is None:
            raise ValueError("Could not set alloy system")
        
        np.random.seed(42)
        data = []
        targets = {
            'additions': [],
            'process': [],
            'performance': [],
            'properties': []
        }
        
        for i in range(n_samples):
            sample = self._generate_sample_data()
            features = self.extract_features(sample)
            
            # Calculate targets
            additions = self._calculate_optimal_additions(sample)
            process_params = self._calculate_optimal_process(sample)
            performance = self._calculate_performance_metrics(sample, additions)
            properties = self._calculate_material_properties(sample, additions)
            
            data.append(features.flatten())
            targets['additions'].append(additions)
            targets['process'].append(process_params)
            targets['performance'].append(performance)
            targets['properties'].append(properties)
        
        X = np.array(data)
        return X, {k: np.array(v) for k, v in targets.items()}
    
    def _generate_sample_data(self) -> Dict:
        """Generate realistic sample data for current alloy system."""
        if self.alloy_system is None:
            raise ValueError("Alloy system must be set before generating sample data")
            
        base_element = self.alloy_system['base_elements'][0]
        temp_range = self.alloy_system['temp_range']
        
        # Generate composition based on alloy type
        composition = self._generate_composition()
        target_composition = self._generate_target_composition(composition)
        
        # Generate process parameters
        sample = {
            "timestamp": datetime.now().isoformat() + "Z",
            "batch_id": np.random.randint(100, 1000),
            "spectrometer": composition,
            "furnace_temp": {
                "zone1": np.random.randint(temp_range[0]-50, temp_range[0]+50),
                "zone2": np.random.randint(temp_range[0]-40, temp_range[0]+60),
                "zone3": np.random.randint(temp_range[0]-45, temp_range[0]+55)
            },
            "dosing": self._generate_dosing(),
            "stirrer": {
                "rpm": np.random.randint(120, 200),
                "torque": np.random.randint(40, 70),
                "time_min": np.random.uniform(6, 15)
            },
            "load_cell": {
                "batch_weight_kg": np.random.randint(1500, 2500)
            },
            "gas_flow": {
                "O2_percent": np.random.uniform(0.05, 0.3),
                "flow_L_per_min": np.random.uniform(3, 8)
            },
            "cooling": {
                "cool_rate_C_per_min": np.random.uniform(10, 25)
            },
            "target_composition": target_composition,
            "historical_data": {
                "previous_iterations": np.random.randint(1, 6),
                "average_energy_consumption_kwh": np.random.uniform(400, 700),
                "last_batch_additions": self._generate_dosing()
            }
        }
        
        return sample
    
    def _generate_composition(self) -> Dict[str, float]:
        """Generate realistic composition for current alloy type."""
        composition = {}
        
        if self.alloy_type == 'steel':
            composition = {
                'Fe': np.random.uniform(85, 95),
                'C': np.random.uniform(0.01, 1.5),
                'Si': np.random.uniform(0.1, 2.5),
                'Mn': np.random.uniform(0.1, 2.0),
                'P': np.random.uniform(0.001, 0.05),
                'S': np.random.uniform(0.001, 0.05),
                'Cr': np.random.uniform(0.1, 18.0),
                'Ni': np.random.uniform(0.1, 12.0),
                'Mo': np.random.uniform(0.01, 3.0),
                'Cu': np.random.uniform(0.01, 2.0)
            }
        elif self.alloy_type == 'aluminum':
            composition = {
                'Al': np.random.uniform(85, 95),
                'Cu': np.random.uniform(0.5, 6.0),
                'Si': np.random.uniform(0.5, 8.0),
                'Mg': np.random.uniform(0.1, 3.0),
                'Zn': np.random.uniform(0.1, 7.0),
                'Fe': np.random.uniform(0.1, 1.0),
                'Mn': np.random.uniform(0.1, 1.5)
            }
        elif self.alloy_type == 'copper':
            composition = {
                'Cu': np.random.uniform(70, 95),
                'Zn': np.random.uniform(1, 25),
                'Sn': np.random.uniform(0.5, 12),
                'Ni': np.random.uniform(0.1, 15),
                'Al': np.random.uniform(0.1, 8),
                'Fe': np.random.uniform(0.1, 3)
            }
        # Add more alloy types as needed
        
        # Normalize to ensure realistic totals
        total = sum(composition.values())
        if total > 100:
            factor = 98 / total  # Leave some room for trace elements
            composition = {k: v * factor for k, v in composition.items()}
        
        return {k: round(v, 2) for k, v in composition.items()}
    
    def _generate_target_composition(self, current: Dict[str, float]) -> Dict[str, float]:
        """Generate target composition with realistic deviations from current."""
        if self.alloy_system is None:
            raise ValueError("Alloy system must be set")
            
        target = {}
        
        for element in current:
            # Add small random variations around current composition
            deviation = np.random.uniform(-0.5, 0.5)
            if element in self.alloy_system['base_elements']:
                # Base elements should not deviate much
                deviation *= 0.5
            
            target[element] = max(0, current[element] + deviation)
        
        return {k: round(v, 2) for k, v in target.items()}
    
    def _generate_dosing(self) -> Dict[str, float]:
        """Generate dosing amounts for addable elements."""
        dosing = {}
        for element in self.addable_elements:
            dosing[f'{element}_added'] = round(np.random.uniform(0, 3), 1)
        return dosing
    
    def _calculate_optimal_additions(self, sample: Dict) -> List[float]:
        """Calculate optimal additions based on alloy-specific rules."""
        if self.alloy_system is None:
            raise ValueError("Alloy system must be set")
            
        additions = []
        current = sample['spectrometer']
        target = sample['target_composition']
        batch_weight = sample['load_cell']['batch_weight_kg']
        density = self.alloy_system['density']
        
        for element in self.addable_elements:
            current_pct = current.get(element, 0.0)
            target_pct = target.get(element, 0.0)
            deviation = target_pct - current_pct
            
            if deviation > 0:
                # Calculate addition with alloy-specific efficiency
                efficiency = self._get_element_efficiency(element)
                addition_kg = (deviation / 100.0) * batch_weight * efficiency
                
                # Apply element-specific constraints
                addition_kg = self._apply_element_constraints(element, addition_kg)
            else:
                addition_kg = 0.0
            
            additions.append(round(addition_kg, 1))
        
        return additions
    
    def _get_element_efficiency(self, element: str) -> float:
        """Get efficiency factor for element addition in current alloy."""
        efficiency_map = {
            'steel': {'Fe': 0.95, 'C': 0.8, 'Si': 0.85, 'Mn': 0.9, 'Cr': 0.92, 'Ni': 0.95, 'Mo': 0.88},
            'aluminum': {'Al': 0.95, 'Cu': 0.85, 'Si': 0.8, 'Mg': 0.75, 'Zn': 0.9},
            'copper': {'Cu': 0.95, 'Zn': 0.9, 'Sn': 0.85, 'Ni': 0.88, 'Al': 0.8}
        }
        
        return efficiency_map.get(self.alloy_type, {}).get(element, 0.85)
    
    def _apply_element_constraints(self, element: str, addition_kg: float) -> float:
        """Apply realistic constraints on element additions."""
        constraints = {
            'steel': {'Fe': 10, 'C': 0.5, 'Si': 2, 'Mn': 3, 'Cr': 5, 'Ni': 5, 'Mo': 2},
            'aluminum': {'Al': 8, 'Cu': 3, 'Si': 2, 'Mg': 1.5, 'Zn': 3},
            'copper': {'Cu': 10, 'Zn': 5, 'Sn': 3, 'Ni': 4, 'Al': 2}
        }
        
        max_addition = constraints.get(self.alloy_type, {}).get(element, 5.0)
        return max(0, min(addition_kg, max_addition))
    
    def _calculate_optimal_process(self, sample: Dict) -> List[float]:
        """Calculate optimal process parameters for current alloy."""
        if self.alloy_system is None:
            raise ValueError("Alloy system must be set")
            
        temp_range = self.alloy_system['temp_range']
        base_temp = (temp_range[0] + temp_range[1]) / 2
        
        # Alloy-specific temperature adjustments
        target = sample['target_composition']
        temp_adjustment = self._calculate_temperature_adjustment(target)
        
        optimal_temps = [
            base_temp + temp_adjustment - 5,
            base_temp + temp_adjustment + 3,
            base_temp + temp_adjustment
        ]
        
        # Stirrer optimization based on alloy viscosity and complexity
        viscosity_factor = self._get_alloy_viscosity_factor(target)
        optimal_rpm = 140 + viscosity_factor * 30
        optimal_time = 8 + viscosity_factor * 4
        
        return optimal_temps + [optimal_rpm, optimal_time]
    
    def _calculate_temperature_adjustment(self, composition: Dict) -> float:
        """Calculate temperature adjustment based on composition."""
        if self.alloy_type == 'steel':
            # Higher C and Cr content need higher temps
            return (composition.get('C', 0) * 10 + composition.get('Cr', 0) * 2)
        elif self.alloy_type == 'aluminum':
            # Higher Cu content needs higher temps
            return composition.get('Cu', 0) * 3
        elif self.alloy_type == 'copper':
            # Higher Zn content needs higher temps
            return composition.get('Zn', 0) * 2
        else:
            return 0
    
    def _get_alloy_viscosity_factor(self, composition: Dict) -> float:
        """Get relative viscosity factor for stirring optimization."""
        if self.alloy_type == 'steel':
            return min(1.0, (composition.get('C', 0) + composition.get('Cr', 0) / 10))
        elif self.alloy_type == 'aluminum':
            return min(1.0, composition.get('Si', 0) / 8)
        else:
            return 0.5  # Default
    
    def _calculate_performance_metrics(self, sample: Dict, additions: List[float]) -> List[float]:
        """Calculate performance metrics (iterations, energy, accuracy)."""
        # Calculate based on total deviation and alloy complexity
        current = sample['spectrometer']
        target = sample['target_composition']
        
        total_deviation = sum(abs(current.get(e, 0) - target.get(e, 0)) 
                            for e in self.addable_elements if e in target)
        
        iterations_saved = max(0, min(4, int(total_deviation / 0.4)))
        energy_saving_pct = iterations_saved * 2.8 + np.random.uniform(-1, 1)
        accuracy_pct = max(90.0, 100.0 - total_deviation * 1.5)
        
        return [iterations_saved, max(0, energy_saving_pct), min(100, accuracy_pct)]
    
    def _calculate_material_properties(self, sample: Dict, additions: List[float]) -> List[float]:
        """Calculate predicted material properties after additions."""
        # Simplified property predictions - in reality these would be complex metallurgical models
        target = sample['target_composition']
        
        if self.alloy_type == 'steel':
            # Estimate hardness, tensile strength, impact toughness
            hardness = 150 + target.get('C', 0) * 200 + target.get('Cr', 0) * 5
            tensile = 400 + target.get('C', 0) * 300 + target.get('Mn', 0) * 50
            toughness = 100 - target.get('C', 0) * 30 + target.get('Ni', 0) * 5
        elif self.alloy_type == 'aluminum':
            # Estimate yield strength, elongation, corrosion resistance
            hardness = 50 + target.get('Cu', 0) * 15 + target.get('Zn', 0) * 8
            tensile = 200 + target.get('Cu', 0) * 40 + target.get('Si', 0) * 20
            toughness = 80 + target.get('Mg', 0) * 15
        else:
            # Generic properties
            hardness = 100 + np.random.uniform(-20, 20)
            tensile = 300 + np.random.uniform(-50, 50)
            toughness = 60 + np.random.uniform(-10, 10)
        
        return [hardness, tensile, toughness]
    
    def fit(self, X: Optional[np.ndarray] = None, y: Optional[Dict[str, np.ndarray]] = None):
        """Train all models."""
        if X is None or y is None:
            logger.info("Generating synthetic training data for %s alloys...", self.alloy_type)
            X, y = self.generate_synthetic_training_data(5000)
        
        self.create_feature_names()
        
        logger.info("Training addition prediction model...")
        self.models['additions'].fit(X, y['additions'])
        
        logger.info("Training process optimization model...")
        self.models['process'].fit(X, y['process'])
        
        logger.info("Training performance prediction model...")
        self.models['performance'].fit(X, y['performance'])
        
        logger.info("Training material properties model...")
        self.models['properties'].fit(X, y['properties'])
        
        self.is_trained = True
        logger.info("All models trained successfully for %s alloys!", self.alloy_type)
    
    def predict_comprehensive(self, input_data: Dict) -> Dict:
        """Make comprehensive predictions for any alloy type."""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Auto-detect alloy type if needed
        if self.alloy_type == 'auto':
            detected_type = self.auto_detect_alloy_type(input_data['spectrometer'])
            self._set_alloy_system(detected_type)
        
        # Extract features
        features = self.extract_features(input_data)
        
        # Make predictions
        additions_pred = self.models['additions'].predict(features)[0]
        process_pred = self.models['process'].predict(features)[0]
        performance_pred = self.models['performance'].predict(features)[0]
        properties_pred = self.models['properties'].predict(features)[0]
        
        # Generate comprehensive output
        output = self._generate_detailed_output(
            input_data, additions_pred, process_pred, performance_pred, properties_pred
        )
        
        return output
    
    def _generate_detailed_output(self, input_data: Dict, additions: np.ndarray, 
                                process: np.ndarray, performance: np.ndarray,
                                properties: np.ndarray) -> Dict:
        """Generate detailed output for any alloy type."""
        
        # Parse predictions
        n_temps = 3
        temps = process[:n_temps]
        stirrer_rpm, stirrer_time = process[n_temps:n_temps+2]
        iterations_saved, energy_saving, accuracy = performance
        hardness, tensile, toughness = properties
        
        # Create additions dict
        predicted_additions = {}
        for i, element in enumerate(self.addable_elements):
            predicted_additions[f'{element}_add_kg'] = round(max(0, additions[i]), 1)
        
        # Current and target compositions
        current = input_data['spectrometer']
        target = input_data['target_composition']
        
        output = {
            "batch_id": input_data['batch_id'],
            "timestamp": datetime.now().isoformat() + "Z",
            "alloy_type": self.alloy_type,
            "predicted_additions": predicted_additions,
            "recommended_stirrer": {
                "rpm": int(stirrer_rpm),
                "time_min": round(stirrer_time, 0)
            },
            "recommended_furnace_temp": {
                "zone1": int(temps[0]),
                "zone2": int(temps[1]),
                "zone3": int(temps[2])
            },
            "iterations_saved": int(max(0, iterations_saved)),
            "estimated_energy_saving_percent": round(max(0, energy_saving), 1),
            "composition_accuracy_percent": round(min(100, accuracy), 1),
            "predicted_properties": {
                "hardness": round(hardness, 1),
                "tensile_strength": round(tensile, 1),
                "toughness": round(toughness, 1)
            },
            "reason_for_deviation": {},
            "impact_analysis": {},
            "how_iterations_saved": "",
            "notes": ""
        }
        
        # Generate reason for deviation
        for element in self.addable_elements:
            current_val = current.get(element, 0)
            target_val = target.get(element, 0)
            deviation = current_val - target_val
            
            if abs(deviation) > 0.1:
                if deviation < 0:
                    output["reason_for_deviation"][element] = f"Current {element} % is lower than target due to insufficient addition in previous iteration"
                else:
                    output["reason_for_deviation"][element] = f"Current {element} % is higher than target due to over-addition in previous cycle"
            else:
                output["reason_for_deviation"][element] = f"{element} is at target; no significant deviation"
        
        # Generate impact analysis based on alloy type
        self._add_alloy_specific_analysis(output, current, target)
        
        # Generate notes
        notes_parts = []
        for element, amount in predicted_additions.items():
            if amount > 0.1:
                notes_parts.append(f"Add {amount}kg {element.replace('_add_kg', '')}")
        
        if notes_parts:
            output["notes"] = ". ".join(notes_parts) + f". Optimize furnace zones and stirring for {self.alloy_type} alloy. Expected composition matches target with high accuracy."
        else:
            output["notes"] = f"No metal additions required for {self.alloy_type} alloy. Optimize process parameters only."
        
        output["how_iterations_saved"] = f"ML predicts exact metal amounts and optimized stirring/temperature for {self.alloy_type} alloy, avoiding trial-and-error additions. Normally {int(iterations_saved) + 1} extra iterations would be needed, ML reduces to 1 actual addition cycle."
        
        return output
    
    def _add_alloy_specific_analysis(self, output: Dict, current: Dict, target: Dict):
        """Add alloy-specific impact analysis."""
        total_deviation = sum(abs(current.get(e, 0) - target.get(e, 0)) 
                            for e in self.addable_elements if e in target)
        
        if self.alloy_type == 'steel':
            if total_deviation > 0.5:
                output["impact_analysis"] = {
                    "mechanical_properties": "Hardness and tensile strength may be affected if composition deviations not corrected",
                    "corrosion_resistance": "Corrosion resistance may be compromised, especially if Cr content is low",
                    "production_delay": f"Without correction, would require {int(output['iterations_saved']) + 1} extra heat treatment cycles"
                }
            else:
                output["impact_analysis"] = {
                    "mechanical_properties": "Steel properties within acceptable range",
                    "corrosion_resistance": "Adequate corrosion resistance maintained", 
                    "production_delay": "Minimal production impact expected"
                }
        elif self.alloy_type == 'aluminum':
            if total_deviation > 0.8:
                output["impact_analysis"] = {
                    "mechanical_properties": "Strength and formability may be affected if Cu/Si content not optimized",
                    "corrosion_resistance": "Corrosion resistance maintained for aluminum alloy",
                    "production_delay": f"Without correction, would require {int(output['iterations_saved']) + 1} extra solution treatment cycles"
                }
            else:
                output["impact_analysis"] = {
                    "mechanical_properties": "Aluminum alloy properties within specification",
                    "corrosion_resistance": "Excellent corrosion resistance maintained",
                    "production_delay": "No significant production delay"
                }
        # Add more alloy-specific analysis as needed
    
    def save_model(self, filepath: str):
        """Save the universal alloy model."""
        model_data = {
            'alloy_type': self.alloy_type,
            'alloy_system': self.alloy_system,
            'models': self.models,
            'feature_names': self.feature_names,
            'elements': self.elements,
            'addable_elements': self.addable_elements,
            'config': self.config
        }
        joblib.dump(model_data, filepath)
        logger.info("Universal alloy model saved to %s", filepath)
    
    def load_model(self, filepath: str):
        """Load the universal alloy model."""
        model_data = joblib.load(filepath)
        self.alloy_type = model_data['alloy_type']
        self.alloy_system = model_data['alloy_system']
        self.models = model_data['models']
        self.feature_names = model_data['feature_names']
        self.elements = model_data['elements']
        self.addable_elements = model_data['addable_elements']
        self.is_trained = True
        logger.info("Universal alloy model loaded from %s", filepath)