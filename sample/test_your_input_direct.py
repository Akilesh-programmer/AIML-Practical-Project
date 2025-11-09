#!/usr/bin/env python3
"""
Direct test of your input data using the Universal Alloy Optimizer (without API)
"""

import sys
import os
import json

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from universal_alloy_optimizer import UniversalAlloyOptimizer

def test_your_input_direct():
    """Test your input data directly with the optimizer."""
    
    print("🧪 TESTING YOUR INPUT DATA (Direct Method)")
    print("="*60)
    
    # Your input data 
    input_data = {
        "timestamp": "2025-09-21T10:30:00Z",
        "batch_id": 101,
        "spectrometer": {
            "Al": 4.5,
            "Cu": 2.0,
            "Si": 4.0,
            "Fe": 88.0,  # High Fe content - this is a steel alloy
            "Mn": 1.5,
            "Zn": 0.5
        },
        "furnace_temp": {
            "zone1": 1248,
            "zone2": 1252,
            "zone3": 1250
        },
        "dosing": {
            "Al_added": 3,
            "Cu_added": 5,
            "Si_added": 1
        },
        "stirrer": {
            "rpm": 150,
            "torque": 50,
            "time_min": 10
        },
        "load_cell": {
            "batch_weight_kg": 2000
        },
        "gas_flow": {
            "O2_percent": 0.1,
            "Ar_percent": 99.9
        },
        # Adding target composition for optimization
        "target_composition": {
            "Fe": 87.0,
            "Al": 5.0,
            "Cu": 2.5,
            "Si": 4.5,
            "Mn": 1.0
        },
        "historical_data": {
            "previous_iterations": 2,
            "average_energy_consumption_kwh": 450
        }
    }
    
    print("📋 YOUR INPUT COMPOSITION:")
    print(f"   Fe: {input_data['spectrometer']['Fe']}% (Steel alloy detected)")
    print(f"   Al: {input_data['spectrometer']['Al']}%") 
    print(f"   Cu: {input_data['spectrometer']['Cu']}%")
    print(f"   Si: {input_data['spectrometer']['Si']}%")
    print(f"   Mn: {input_data['spectrometer']['Mn']}%")
    print(f"   Zn: {input_data['spectrometer']['Zn']}%")
    print(f"   Batch Weight: {input_data['load_cell']['batch_weight_kg']} kg")
    print(f"   Batch ID: {input_data['batch_id']}")
    
    print("\n📋 TARGET COMPOSITION:")
    target = input_data['target_composition']
    print(f"   Fe: {target['Fe']}%")
    print(f"   Al: {target['Al']}%")
    print(f"   Cu: {target['Cu']}%")
    print(f"   Si: {target['Si']}%")
    print(f"   Mn: {target['Mn']}%")
    
    print(f"\n🏭 PROCESS CONDITIONS:")
    print(f"   Furnace Zones: {input_data['furnace_temp']['zone1']}°C, {input_data['furnace_temp']['zone2']}°C, {input_data['furnace_temp']['zone3']}°C")
    print(f"   Stirrer: {input_data['stirrer']['rpm']} RPM for {input_data['stirrer']['time_min']} min")
    print(f"   Gas Flow: {input_data['gas_flow']['O2_percent']}% O2, {input_data['gas_flow']['Ar_percent']}% Ar")
    
    try:
        # Create steel alloy optimizer and load the model
        print(f"\n🔄 Loading steel alloy model...")
        optimizer = UniversalAlloyOptimizer(alloy_type='steel')
        optimizer.load_model('models/steel_alloy_model.pkl')
        
        # Get prediction
        print("🔄 Generating prediction...")
        result = optimizer.predict_comprehensive(input_data)
        
        print("\n" + "="*60)
        print("🎯 PREDICTED OUTPUT RESULTS:")
        print("="*60)
        print(f"   Alloy Type: {result['alloy_type']}")
        print(f"   Composition Accuracy: {result['composition_accuracy_percent']:.1f}%")
        print(f"   Iterations Saved: {result['iterations_saved']}")
        print(f"   Energy Saving: {result['estimated_energy_saving_percent']:.1f}%")
        
        print("\n🧪 RECOMMENDED METAL ADDITIONS:")
        additions = result['predicted_additions']
        total_additions = 0
        for element, amount in additions.items():
            if amount > 0.1:
                print(f"   {element}: {amount:.1f} kg")
                total_additions += amount
        
        if total_additions == 0:
            print("   No additional metals required!")
        else:
            print(f"   Total additions: {total_additions:.1f} kg")
        
        print("\n⚙️ OPTIMIZED PROCESS PARAMETERS:")
        stirrer = result['recommended_stirrer']
        temps = result['recommended_furnace_temp']
        print(f"   Stirrer Speed: {stirrer['rpm']} RPM")
        print(f"   Stirring Time: {stirrer['time_min']} minutes")
        print(f"   Furnace Zone 1: {temps['zone1']}°C")
        print(f"   Furnace Zone 2: {temps['zone2']}°C") 
        print(f"   Furnace Zone 3: {temps['zone3']}°C")
        
        print("\n📊 PREDICTED MATERIAL PROPERTIES:")
        props = result['predicted_properties']
        print(f"   Hardness: {props['hardness']:.1f} HV")
        print(f"   Tensile Strength: {props['tensile_strength']:.1f} MPa")
        print(f"   Toughness: {props['toughness']:.1f} J")
        
        print(f"\n📝 OPTIMIZATION NOTES:")
        print(f"   {result['notes']}")
        
        # Save complete result
        with open('your_input_prediction.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n💾 Complete prediction saved to: your_input_prediction.json")
        print(f"\n✅ PREDICTION COMPLETED SUCCESSFULLY!")
        
        return result
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return None

if __name__ == "__main__":
    result = test_your_input_direct()
    if result:
        print(f"\n🎉 Your steel alloy optimization is complete!")
    else:
        print(f"\n💥 Failed to process your input data.")