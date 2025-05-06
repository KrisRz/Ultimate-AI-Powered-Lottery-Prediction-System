"""Model deployment utilities."""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import json
import os
from pathlib import Path
import shutil
from datetime import datetime
import joblib
import pickle
from .model_utils import save_model, load_model, evaluate_model
from .model_validation import validate_model_performance

logger = logging.getLogger(__name__)

class ModelDeployer:
    """Deploy and manage model versions."""
    
    def __init__(self, model: Any, model_name: str, deployment_dir: str):
        """Initialize model deployer."""
        self.model = model
        self.model_name = model_name
        self.deployment_dir = Path(deployment_dir)
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize version tracking
        self.version_file = self.deployment_dir / "versions.json"
        self.versions = self._load_versions()
    
    def _load_versions(self) -> Dict[str, Any]:
        """Load version information from disk."""
        if self.version_file.exists():
            with open(self.version_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_versions(self) -> None:
        """Save version information to disk."""
        with open(self.version_file, 'w') as f:
            json.dump(self.versions, f, indent=4)
    
    def deploy_model(self, version: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Deploy a new model version."""
        # Create version directory
        version_dir = self.deployment_dir / version
        version_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = version_dir / "model"
        save_model(self.model, str(model_path), self.model_name)
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'deployment_time': datetime.now().isoformat(),
            'model_name': self.model_name,
            'version': version
        })
        
        with open(version_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Update version tracking
        self.versions[version] = {
            'path': str(version_dir),
            'metadata': metadata,
            'status': 'active'
        }
        self._save_versions()
        
        return str(version_dir)
    
    def load_version(self, version: str) -> Tuple[Any, Dict[str, Any]]:
        """Load a specific model version."""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        
        version_dir = Path(self.versions[version]['path'])
        model_path = version_dir / "model"
        
        # Load model
        model = load_model(str(model_path))
        
        # Load metadata
        with open(version_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        return model, metadata
    
    def validate_deployment(self, version: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Validate a deployed model version."""
        model, metadata = self.load_version(version)
        
        # Evaluate performance
        metrics = evaluate_model(model, X, y)
        
        # Validate model
        validation_results = validate_model_performance(model, X, y)
        
        return {
            'version': version,
            'metrics': metrics,
            'validation': validation_results,
            'metadata': metadata
        }
    
    def rollback_version(self, version: str) -> None:
        """Rollback to a previous version."""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        
        # Update version status
        self.versions[version]['status'] = 'active'
        for v in self.versions:
            if v != version:
                self.versions[v]['status'] = 'inactive'
        
        self._save_versions()
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """List all deployed versions."""
        return [
            {
                'version': v,
                'status': info['status'],
                'deployment_time': info['metadata']['deployment_time']
            }
            for v, info in self.versions.items()
        ]
    
    def archive_version(self, version: str) -> None:
        """Archive a model version."""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        
        # Create archive directory
        archive_dir = self.deployment_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        
        # Move version to archive
        version_dir = Path(self.versions[version]['path'])
        archive_path = archive_dir / f"{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(version_dir), str(archive_path))
        
        # Update version tracking
        del self.versions[version]
        self._save_versions()
    
    def export_model(self, version: str, export_format: str = 'tf') -> str:
        """Export model in specified format."""
        model, _ = self.load_version(version)
        
        export_dir = self.deployment_dir / "exports" / version
        export_dir.mkdir(parents=True, exist_ok=True)
        
        if export_format == 'tf':
            # Export as TensorFlow SavedModel
            tf.saved_model.save(model, str(export_dir))
        elif export_format == 'onnx':
            # Export as ONNX
            import tf2onnx
            model_proto, _ = tf2onnx.convert.from_keras(model)
            with open(export_dir / "model.onnx", 'wb') as f:
                f.write(model_proto.SerializeToString())
        elif export_format == 'joblib':
            # Export as joblib
            joblib.dump(model, export_dir / "model.joblib")
        elif export_format == 'pickle':
            # Export as pickle
            with open(export_dir / "model.pkl", 'wb') as f:
                pickle.dump(model, f)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        return str(export_dir)
    
    def create_deployment_package(self, version: str) -> str:
        """Create a deployment package for a model version."""
        model, metadata = self.load_version(version)
        
        # Create package directory
        package_dir = self.deployment_dir / "packages" / version
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model and metadata
        version_dir = Path(self.versions[version]['path'])
        shutil.copytree(version_dir, package_dir / "model", dirs_exist_ok=True)
        
        # Create requirements file
        requirements = {
            'tensorflow': tf.__version__,
            'numpy': np.__version__,
            'scikit-learn': '>=1.0.0'
        }
        
        with open(package_dir / "requirements.txt", 'w') as f:
            for package, version in requirements.items():
                f.write(f"{package}{version}\n")
        
        # Create deployment script
        deployment_script = f"""#!/bin/bash
# Model deployment script for {self.model_name} version {version}

# Install requirements
pip install -r requirements.txt

# Load and validate model
python -c "
import joblib
import json
import numpy as np
from pathlib import Path

# Load model
model_path = Path('model')
model = joblib.load(model_path / 'model.joblib')

# Load metadata
with open(model_path / 'metadata.json', 'r') as f:
    metadata = json.load(f)

print(f'Successfully loaded model {metadata['model_name']} version {metadata['version']}')
"
"""
        
        with open(package_dir / "deploy.sh", 'w') as f:
            f.write(deployment_script)
        os.chmod(package_dir / "deploy.sh", 0o755)
        
        # Create package
        package_path = self.deployment_dir / f"{self.model_name}_{version}.tar.gz"
        shutil.make_archive(
            str(package_path).replace('.tar.gz', ''),
            'gztar',
            package_dir
        )
        
        return str(package_path)
    
    def validate_deployment_package(self, package_path: str) -> Dict[str, Any]:
        """Validate a deployment package."""
        # Extract package
        extract_dir = self.deployment_dir / "temp_validation"
        extract_dir.mkdir(exist_ok=True)
        
        shutil.unpack_archive(package_path, extract_dir)
        
        # Check required files
        required_files = ['model/model.joblib', 'model/metadata.json', 'requirements.txt', 'deploy.sh']
        missing_files = [f for f in required_files if not (extract_dir / f).exists()]
        
        if missing_files:
            return {
                'valid': False,
                'errors': [f'Missing required files: {missing_files}']
            }
        
        # Load and validate model
        try:
            model = joblib.load(extract_dir / "model/model.joblib")
            with open(extract_dir / "model/metadata.json", 'r') as f:
                metadata = json.load(f)
            
            # Clean up
            shutil.rmtree(extract_dir)
            
            return {
                'valid': True,
                'metadata': metadata
            }
        except Exception as e:
            # Clean up
            shutil.rmtree(extract_dir)
            
            return {
                'valid': False,
                'errors': [str(e)]
            } 