"""
Training script for Universal Multi-Metal Alloy Optimizer
Trains models for all supported alloy types with proper validation
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime
from universal_alloy_optimizer import UniversalAlloyOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_all_alloy_models():
    """Train models for all supported alloy types."""
    alloy_types = ['steel', 'aluminum', 'copper', 'titanium', 'nickel', 'magnesium']
    models = {}
    
    print("🔥 Universal Multi-Metal Alloy Optimizer Training")
    print("=" * 60)
    
    for alloy_type in alloy_types:
        print(f"\n🏭 Training {alloy_type.upper()} alloy model...")
        
        try:
            # Initialize model
            model = UniversalAlloyOptimizer(alloy_type=alloy_type)
            
            # Generate training data
            logger.info("Generating synthetic training data for %s...", alloy_type)
            X, y = model.generate_synthetic_training_data(n_samples=8000)
            
            # Train model
            logger.info("Training %s alloy model...", alloy_type)
            model.fit(X, y)
            
            # Save model
            model_path = f"models/{alloy_type}_alloy_model.pkl"
            Path("models").mkdir(exist_ok=True)
            model.save_model(model_path)
            
            models[alloy_type] = model
            
            print(f"✅ {alloy_type.capitalize()} model trained and saved successfully!")
            
            # Print model info
            print(f"   - Training samples: {X.shape[0]}")
            print(f"   - Feature count: {X.shape[1]}")
            print(f"   - Addable elements: {', '.join(model.addable_elements)}")
            print(f"   - Model saved: {model_path}")
            
        except Exception as e:
            logger.error("Failed to train %s model: %s", alloy_type, str(e))
            print(f"❌ Failed to train {alloy_type} model: {str(e)}")
    
    return models

def test_model_predictions(models):
    """Test model predictions with sample data."""
    print(f"\n🧪 Testing Model Predictions")
    print("=" * 60)
    
    # Test data for different alloys
    test_cases = {
        'steel': {
            "timestamp": datetime.now().isoformat() + "Z",
            "batch_id": 1001,
            "spectrometer": {
                "Fe": 92.5, "C": 0.45, "Si": 0.8, "Mn": 1.2,
                "Cr": 2.5, "Ni": 1.5, "Mo": 0.3
            },
            "furnace_temp": {"zone1": 1520, "zone2": 1540, "zone3": 1530},
            "stirrer": {"rpm": 150, "torque": 55, "time_min": 10.0},
            "load_cell": {"batch_weight_kg": 2000},
            "target_composition": {
                "Fe": 91.8, "C": 0.50, "Si": 0.9, "Mn": 1.3,
                "Cr": 3.0, "Ni": 2.0, "Mo": 0.4
            }
        },
        'aluminum': {
            "timestamp": datetime.now().isoformat() + "Z",
            "batch_id": 2001,
            "spectrometer": {
                "Al": 89.5, "Cu": 4.2, "Si": 2.1, "Mg": 1.8, "Zn": 1.5
            },
            "furnace_temp": {"zone1": 1220, "zone2": 1240, "zone3": 1230},
            "stirrer": {"rpm": 140, "torque": 45, "time_min": 8.0},
            "load_cell": {"batch_weight_kg": 1500},
            "target_composition": {
                "Al": 88.0, "Cu": 5.0, "Si": 2.5, "Mg": 2.2, "Zn": 1.8
            }
        },
        'copper': {
            "timestamp": datetime.now().isoformat() + "Z",
            "batch_id": 3001,
            "spectrometer": {
                "Cu": 78.5, "Zn": 18.0, "Sn": 2.0, "Ni": 1.2
            },
            "furnace_temp": {"zone1": 1080, "zone2": 1100, "zone3": 1090},
            "stirrer": {"rpm": 130, "torque": 40, "time_min": 7.0},
            "load_cell": {"batch_weight_kg": 1200},
            "target_composition": {
                "Cu": 76.0, "Zn": 20.0, "Sn": 2.5, "Ni": 1.5
            }
        }
    }
    
    test_results = {}
    
    for alloy_type, test_data in test_cases.items():
        if alloy_type in models:
            try:
                model = models[alloy_type]
                result = model.predict_comprehensive(test_data)
                test_results[alloy_type] = result
                
                print(f"\n🔬 {alloy_type.upper()} Test Results:")
                print(f"   Batch ID: {result['batch_id']}")
                print(f"   Accuracy: {result['composition_accuracy_percent']:.1f}%")
                print(f"   Iterations Saved: {result['iterations_saved']}")
                print(f"   Energy Saving: {result['estimated_energy_saving_percent']:.1f}%")
                
                # Show predicted additions
                additions = result['predicted_additions']
                non_zero_additions = {k: v for k, v in additions.items() if v > 0.1}
                if non_zero_additions:
                    print(f"   Additions: {', '.join(f'{k}:{v}kg' for k, v in non_zero_additions.items())}")
                else:
                    print(f"   Additions: None required")
                
                print("✅ Test passed!")
                
            except Exception as e:
                logger.error("Test failed for %s: %s", alloy_type, str(e))
                print(f"❌ Test failed for {alloy_type}: {str(e)}")
    
    return test_results

def validate_model_consistency(models):
    """Validate model consistency and performance."""
    print(f"\n🎯 Model Validation")
    print("=" * 60)
    
    validation_results = {}
    
    for alloy_type, model in models.items():
        try:
            # Generate validation data
            X_val, y_val = model.generate_synthetic_training_data(n_samples=1000)
            
            # Make predictions
            val_features = []
            for i in range(min(100, X_val.shape[0])):  # Test first 100 samples
                sample_data = model._generate_sample_data()
                features = model.extract_features(sample_data)
                val_features.append(features.flatten())
            
            val_features = np.array(val_features)
            
            # Test all sub-models
            additions_pred = model.models['additions'].predict(val_features)
            process_pred = model.models['process'].predict(val_features)
            performance_pred = model.models['performance'].predict(val_features)
            properties_pred = model.models['properties'].predict(val_features)
            
            # Basic validation checks
            valid_additions = np.all(additions_pred >= 0)  # No negative additions
            valid_temps = np.all((process_pred[:, :3] >= 500) & (process_pred[:, :3] <= 2000))  # Reasonable temperatures
            valid_accuracy = np.all((performance_pred[:, 2] >= 0) & (performance_pred[:, 2] <= 100))  # Accuracy 0-100%
            
            validation_results[alloy_type] = {
                'valid_additions': valid_additions,
                'valid_temperatures': valid_temps,
                'valid_accuracy': valid_accuracy,
                'sample_count': val_features.shape[0],
                'feature_count': val_features.shape[1]
            }
            
            status = "✅ PASS" if all([valid_additions, valid_temps, valid_accuracy]) else "❌ FAIL"
            print(f"{alloy_type.capitalize()}: {status}")
            print(f"   Samples: {val_features.shape[0]}, Features: {val_features.shape[1]}")
            print(f"   Valid additions: {valid_additions}")
            print(f"   Valid temperatures: {valid_temps}")
            print(f"   Valid accuracy: {valid_accuracy}")
            
        except Exception as e:
            logger.error("Validation failed for %s: %s", alloy_type, str(e))
            print(f"{alloy_type.capitalize()}: ❌ FAIL - {str(e)}")
            validation_results[alloy_type] = {'error': str(e)}
    
    return validation_results

def save_training_report(models, test_results, validation_results):
    """Save comprehensive training report."""
    report = {
        'training_timestamp': datetime.now().isoformat(),
        'models_trained': list(models.keys()),
        'model_info': {},
        'test_results': test_results,
        'validation_results': validation_results
    }
    
    for alloy_type, model in models.items():
        report['model_info'][alloy_type] = {
            'alloy_system': model.alloy_system,
            'elements': model.elements,
            'addable_elements': model.addable_elements,
            'feature_count': len(model.feature_names) if model.feature_names else 0,
            'is_trained': model.is_trained
        }
    
    # Save report
    report_path = f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📊 Training report saved: {report_path}")
    return report

def main():
    """Main training function."""
    print("🚀 Starting Universal Multi-Metal Alloy Optimizer Training")
    start_time = datetime.now()
    
    try:
        # Train all models
        models = train_all_alloy_models()
        
        if not models:
            print("❌ No models were trained successfully!")
            return
        
        # Test predictions
        test_results = test_model_predictions(models)
        
        # Validate models
        validation_results = validate_model_consistency(models)
        
        # Save comprehensive report
        report = save_training_report(models, test_results, validation_results)
        
        # Summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n🎉 Training Complete!")
        print("=" * 60)
        print(f"Models trained: {len(models)}/{len(['steel', 'aluminum', 'copper', 'titanium', 'nickel', 'magnesium'])}")
        print(f"Training time: {duration}")
        print(f"Models directory: ./models/")
        
        successful_validations = sum(1 for v in validation_results.values() 
                                   if isinstance(v, dict) and v.get('valid_additions') and 
                                      v.get('valid_temperatures') and v.get('valid_accuracy'))
        print(f"Validation passed: {successful_validations}/{len(models)}")
        
        if successful_validations == len(models):
            print("✅ All models are ready for production!")
            print("\nTo start the API server, run:")
            print("python universal_alloy_api.py")
        else:
            print("⚠️  Some models failed validation. Check logs for details.")
        
    except Exception as e:
        logger.error("Training failed: %s", str(e))
        print(f"❌ Training failed: {str(e)}")

if __name__ == "__main__":
    main()