"""
Quick Universal Multi-Metal Alloy Optimizer API
Loads pre-trained models for fast startup
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
import uvicorn
import json
import logging
from datetime import datetime
from universal_alloy_optimizer import UniversalAlloyOptimizer
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Universal Multi-Metal Alloy Optimizer API",
    description="Fast alloy composition optimization for steel, aluminum, copper, titanium, and other alloys",
    version="1.0.0"
)

# Enable CORS (allow all origins; adjust in production for security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consider restricting to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
models = {}

# Pydantic models (same as before)
class SpectrometerData(BaseModel):
    """Spectrometer composition data - dynamically handles any elements"""
    Fe: Optional[float] = Field(None, ge=0, le=100, description="Iron percentage")
    C: Optional[float] = Field(None, ge=0, le=5, description="Carbon percentage")
    Si: Optional[float] = Field(None, ge=0, le=10, description="Silicon percentage")
    Mn: Optional[float] = Field(None, ge=0, le=5, description="Manganese percentage")
    P: Optional[float] = Field(None, ge=0, le=1, description="Phosphorus percentage")
    S: Optional[float] = Field(None, ge=0, le=1, description="Sulfur percentage")
    Cr: Optional[float] = Field(None, ge=0, le=30, description="Chromium percentage")
    Ni: Optional[float] = Field(None, ge=0, le=50, description="Nickel percentage")
    Mo: Optional[float] = Field(None, ge=0, le=10, description="Molybdenum percentage")
    Cu: Optional[float] = Field(None, ge=0, le=10, description="Copper percentage")
    Al: Optional[float] = Field(None, ge=0, le=10, description="Aluminum percentage")
    V: Optional[float] = Field(None, ge=0, le=5, description="Vanadium percentage")
    W: Optional[float] = Field(None, ge=0, le=10, description="Tungsten percentage")
    Co: Optional[float] = Field(None, ge=0, le=20, description="Cobalt percentage")
    Mg: Optional[float] = Field(None, ge=0, le=10, description="Magnesium percentage")
    Zn: Optional[float] = Field(None, ge=0, le=15, description="Zinc percentage")
    Ti: Optional[float] = Field(None, ge=0, le=5, description="Titanium percentage")
    Li: Optional[float] = Field(None, ge=0, le=3, description="Lithium percentage")
    Sn: Optional[float] = Field(None, ge=0, le=20, description="Tin percentage")
    Pb: Optional[float] = Field(None, ge=0, le=10, description="Lead percentage")
    Be: Optional[float] = Field(None, ge=0, le=5, description="Beryllium percentage")
    Zr: Optional[float] = Field(None, ge=0, le=10, description="Zirconium percentage")
    Nb: Optional[float] = Field(None, ge=0, le=10, description="Niobium percentage")
    Ca: Optional[float] = Field(None, ge=0, le=5, description="Calcium percentage")
    Y: Optional[float] = Field(None, ge=0, le=5, description="Yttrium percentage")
    Nd: Optional[float] = Field(None, ge=0, le=5, description="Neodymium percentage")
    
    def dict(self, *args, **kwargs):
        """Override dict to remove None values"""
        d = super().dict(*args, **kwargs)
        return {k: v for k, v in d.items() if v is not None}

class FurnaceTemp(BaseModel):
    zone1: int = Field(..., ge=500, le=2000, description="Zone 1 temperature in Celsius")
    zone2: int = Field(..., ge=500, le=2000, description="Zone 2 temperature in Celsius") 
    zone3: int = Field(..., ge=500, le=2000, description="Zone 3 temperature in Celsius")

class DosingData(BaseModel):
    """Dosing data - handles any addable elements"""
    Fe_added: Optional[float] = Field(0, ge=0, le=20, description="Iron added in kg")
    C_added: Optional[float] = Field(0, ge=0, le=2, description="Carbon added in kg")
    Si_added: Optional[float] = Field(0, ge=0, le=5, description="Silicon added in kg")
    Mn_added: Optional[float] = Field(0, ge=0, le=5, description="Manganese added in kg")
    Cr_added: Optional[float] = Field(0, ge=0, le=5, description="Chromium added in kg")
    Ni_added: Optional[float] = Field(0, ge=0, le=5, description="Nickel added in kg")
    Mo_added: Optional[float] = Field(0, ge=0, le=3, description="Molybdenum added in kg")
    Cu_added: Optional[float] = Field(0, ge=0, le=5, description="Copper added in kg")
    Al_added: Optional[float] = Field(0, ge=0, le=5, description="Aluminum added in kg")
    Mg_added: Optional[float] = Field(0, ge=0, le=3, description="Magnesium added in kg")
    Zn_added: Optional[float] = Field(0, ge=0, le=5, description="Zinc added in kg")
    Ti_added: Optional[float] = Field(0, ge=0, le=3, description="Titanium added in kg")
    Sn_added: Optional[float] = Field(0, ge=0, le=3, description="Tin added in kg")
    V_added: Optional[float] = Field(0, ge=0, le=2, description="Vanadium added in kg")

class StirerData(BaseModel):
    rpm: int = Field(..., ge=100, le=250, description="Stirrer RPM")
    torque: Optional[int] = Field(50, ge=20, le=100, description="Stirrer torque")
    time_min: float = Field(..., ge=5, le=20, description="Stirring time in minutes")

class LoadCellData(BaseModel):
    batch_weight_kg: int = Field(..., ge=500, le=5000, description="Batch weight in kg")

class GasFlowData(BaseModel):
    O2_percent: Optional[float] = Field(0.15, ge=0.05, le=0.5, description="Oxygen percentage")
    flow_L_per_min: Optional[float] = Field(5.0, ge=1, le=15, description="Gas flow rate L/min")

class CoolingData(BaseModel):
    cool_rate_C_per_min: Optional[float] = Field(15.0, ge=5, le=50, description="Cooling rate C/min")

class HistoricalData(BaseModel):
    previous_iterations: Optional[int] = Field(0, ge=0, le=10, description="Previous iterations")
    average_energy_consumption_kwh: Optional[float] = Field(500, ge=100, le=1000, description="Average energy consumption")
    last_batch_additions: Optional[DosingData] = Field(None, description="Last batch additions")

class AlloyOptimizationRequest(BaseModel):
    timestamp: str = Field(..., description="Request timestamp in ISO format")
    batch_id: int = Field(..., ge=1, le=10000, description="Batch ID")
    alloy_type: Optional[str] = Field('auto', description="Alloy type: steel, aluminum, copper, titanium, nickel, magnesium, or auto")
    spectrometer: SpectrometerData = Field(..., description="Current composition from spectrometer")
    furnace_temp: FurnaceTemp = Field(..., description="Current furnace temperatures")
    dosing: Optional[DosingData] = Field(None, description="Previous dosing amounts")
    stirrer: StirerData = Field(..., description="Stirrer parameters")
    load_cell: LoadCellData = Field(..., description="Load cell data")
    gas_flow: Optional[GasFlowData] = Field(None, description="Gas flow parameters")
    cooling: Optional[CoolingData] = Field(None, description="Cooling parameters")
    target_composition: SpectrometerData = Field(..., description="Target composition")
    historical_data: Optional[HistoricalData] = Field(None, description="Historical data")

# Response models (simplified)
class AlloyOptimizationResponse(BaseModel):
    batch_id: int
    timestamp: str
    alloy_type: str
    predicted_additions: Dict[str, float]
    recommended_stirrer: Dict[str, Union[int, float]]
    recommended_furnace_temp: Dict[str, int]
    iterations_saved: int
    estimated_energy_saving_percent: float
    composition_accuracy_percent: float
    predicted_properties: Dict[str, float]
    reason_for_deviation: Dict[str, str]
    impact_analysis: Dict[str, str]
    how_iterations_saved: str
    notes: str

def load_trained_models():
    """Load pre-trained models for fast startup."""
    alloy_types = ['steel', 'aluminum', 'copper', 'titanium', 'nickel', 'magnesium']
    models_dir = Path("models")
    
    if not models_dir.exists():
        raise FileNotFoundError("Models directory not found. Please run training first.")
    
    for alloy_type in alloy_types:
        model_path = models_dir / f"{alloy_type}_alloy_model.pkl"
        if model_path.exists():
            logger.info("Loading %s alloy model...", alloy_type)
            model = UniversalAlloyOptimizer(alloy_type=alloy_type)
            model.load_model(str(model_path))
            models[alloy_type] = model
            logger.info("%s alloy model loaded successfully!", alloy_type)
        else:
            logger.warning("Model file not found: %s", model_path)
    
    # Also create an auto-detection model
    if 'steel' in models:
        models['auto'] = models['steel']  # Use steel as default for auto-detection
    
    logger.info("All available models loaded: %s", list(models.keys()))

def get_model(alloy_type: str) -> Optional[UniversalAlloyOptimizer]:
    """Get the appropriate model for the alloy type."""
    if alloy_type in models:
        return models[alloy_type]
    else:
        # Fallback to auto-detection
        logger.warning("Unknown alloy type %s, using auto-detection", alloy_type)
        return models.get('auto') or models.get('steel')

def prepare_input_data(request: AlloyOptimizationRequest) -> Dict:
    """Convert Pydantic request to dict format expected by optimizer."""
    spectrometer_dict = request.spectrometer.dict()
    target_dict = request.target_composition.dict()
    
    dosing_dict = {}
    if request.dosing:
        dosing_data = request.dosing.dict()
        for k, v in dosing_data.items():
            if v is not None and v > 0:
                dosing_dict[k] = v
    
    historical_dict = {}
    if request.historical_data:
        historical_dict = {
            "previous_iterations": request.historical_data.previous_iterations or 0,
            "average_energy_consumption_kwh": request.historical_data.average_energy_consumption_kwh or 500,
            "last_batch_additions": request.historical_data.last_batch_additions.dict() if request.historical_data.last_batch_additions else {}
        }
    
    input_data = {
        "timestamp": request.timestamp,
        "batch_id": request.batch_id,
        "spectrometer": spectrometer_dict,
        "furnace_temp": {
            "zone1": request.furnace_temp.zone1,
            "zone2": request.furnace_temp.zone2,
            "zone3": request.furnace_temp.zone3
        },
        "dosing": dosing_dict,
        "stirrer": {
            "rpm": request.stirrer.rpm,
            "torque": request.stirrer.torque or 50,
            "time_min": request.stirrer.time_min
        },
        "load_cell": {
            "batch_weight_kg": request.load_cell.batch_weight_kg
        },
        "gas_flow": {
            "O2_percent": request.gas_flow.O2_percent if request.gas_flow else 0.15,
            "flow_L_per_min": request.gas_flow.flow_L_per_min if request.gas_flow else 5.0
        },
        "cooling": {
            "cool_rate_C_per_min": request.cooling.cool_rate_C_per_min if request.cooling else 15.0
        },
        "target_composition": target_dict,
        "historical_data": historical_dict
    }
    
    return input_data

# Initialize models on startup
try:
    logger.info("Starting Quick Universal Multi-Metal Alloy Optimizer API...")
    load_trained_models()
    logger.info("API ready with %d alloy models!", len(models))
except Exception as e:
    logger.error("Failed to load models: %s", str(e))
    models = {}

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Quick Universal Multi-Metal Alloy Optimizer API",
        "version": "1.0.0",
        "loaded_models": list(models.keys()),
        "status": "ready" if models else "no models loaded"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if models else "no models",
        "models_loaded": len(models),
        "available_alloys": list(models.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/optimize", response_model=AlloyOptimizationResponse)
async def optimize_alloy(request: AlloyOptimizationRequest):
    """
    Main alloy optimization endpoint.
    Supports all metal types with automatic detection or manual specification.
    """
    try:
        if not models:
            raise HTTPException(status_code=503, detail="No models loaded. Please train models first.")
        
        logger.info("Processing optimization request for batch %d", request.batch_id)
        
        # Prepare input data
        input_data = prepare_input_data(request)
        
        # Get appropriate model
        model = get_model(request.alloy_type or 'auto')
        if not model:
            raise HTTPException(status_code=400, detail="No suitable model found")
        
        # Make prediction
        result = model.predict_comprehensive(input_data)
        
        # Convert to response format
        response = AlloyOptimizationResponse(**result)
        
        logger.info("Optimization completed for batch %d, alloy type: %s", 
                   request.batch_id, result["alloy_type"])
        
        return response
        
    except Exception as e:
        logger.error("Error processing optimization request: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001,  # Different port to avoid conflicts
        log_level="info"
    )