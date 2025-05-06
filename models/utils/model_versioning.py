"""Model versioning utilities."""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import json
import os
from pathlib import Path
import shutil
from datetime import datetime
import hashlib
import git
from .model_utils import save_model, load_model, evaluate_model
from .model_validation import validate_model_performance

logger = logging.getLogger(__name__)

class ModelVersioner:
    """Manage model versions and track changes."""
    
    def __init__(self, model: Any, model_name: str, versioning_dir: str):
        """Initialize model versioner."""
        self.model = model
        self.model_name = model_name
        self.versioning_dir = Path(versioning_dir)
        self.versioning_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize version tracking
        self.version_file = self.versioning_dir / "versions.json"
        self.versions = self._load_versions()
        
        # Initialize git repository if not exists
        self.repo_path = self.versioning_dir / ".git"
        if not self.repo_path.exists():
            self._init_git_repo()
    
    def _init_git_repo(self) -> None:
        """Initialize git repository for versioning."""
        repo = git.Repo.init(str(self.versioning_dir))
        
        # Create .gitignore
        gitignore = """# Model versioning
*.joblib
*.h5
*.pkl
*.onnx
__pycache__/
*.pyc
.DS_Store
"""
        with open(self.versioning_dir / ".gitignore", 'w') as f:
            f.write(gitignore)
        
        repo.index.add([".gitignore"])
        repo.index.commit("Initial commit")
    
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
        
        # Commit version changes
        repo = git.Repo(str(self.versioning_dir))
        repo.index.add([str(self.version_file)])
        repo.index.commit(f"Update version tracking for {self.model_name}")
    
    def _calculate_model_hash(self, model: Any) -> str:
        """Calculate hash of model parameters."""
        if isinstance(model, tf.keras.Model):
            weights = model.get_weights()
            weights_str = str(weights)
        else:
            weights_str = str(model.get_params())
        
        return hashlib.sha256(weights_str.encode()).hexdigest()
    
    def create_version(self, version: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new model version."""
        # Create version directory
        version_dir = self.versioning_dir / version
        version_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = version_dir / "model"
        save_model(self.model, str(model_path), self.model_name)
        
        # Calculate model hash
        model_hash = self._calculate_model_hash(self.model)
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'creation_time': datetime.now().isoformat(),
            'model_name': self.model_name,
            'version': version,
            'model_hash': model_hash
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
    
    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two model versions."""
        model1, metadata1 = self.load_version(version1)
        model2, metadata2 = self.load_version(version2)
        
        # Compare model hashes
        hash1 = metadata1['model_hash']
        hash2 = metadata2['model_hash']
        
        # Compare model parameters
        if isinstance(model1, tf.keras.Model) and isinstance(model2, tf.keras.Model):
            weights1 = model1.get_weights()
            weights2 = model2.get_weights()
            param_diff = {
                'layers': len(weights1) == len(weights2),
                'shapes': [
                    w1.shape == w2.shape
                    for w1, w2 in zip(weights1, weights2)
                ]
            }
        else:
            params1 = model1.get_params()
            params2 = model2.get_params()
            param_diff = {
                'keys': set(params1.keys()) == set(params2.keys()),
                'values': {
                    k: params1[k] == params2[k]
                    for k in params1.keys()
                    if k in params2
                }
            }
        
        return {
            'versions': {
                'version1': version1,
                'version2': version2
            },
            'hashes': {
                'version1': hash1,
                'version2': hash2,
                'equal': hash1 == hash2
            },
            'parameters': param_diff,
            'metadata': {
                'version1': metadata1,
                'version2': metadata2
            }
        }
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """List all model versions."""
        return [
            {
                'version': v,
                'status': info['status'],
                'creation_time': info['metadata']['creation_time'],
                'model_hash': info['metadata']['model_hash']
            }
            for v, info in self.versions.items()
        ]
    
    def tag_version(self, version: str, tag: str) -> None:
        """Tag a specific model version."""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        
        # Update version metadata
        self.versions[version]['metadata']['tags'] = self.versions[version]['metadata'].get('tags', [])
        if tag not in self.versions[version]['metadata']['tags']:
            self.versions[version]['metadata']['tags'].append(tag)
        
        self._save_versions()
    
    def get_version_by_tag(self, tag: str) -> Optional[str]:
        """Get version by tag."""
        for version, info in self.versions.items():
            if tag in info['metadata'].get('tags', []):
                return version
        return None
    
    def archive_version(self, version: str) -> None:
        """Archive a model version."""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        
        # Create archive directory
        archive_dir = self.versioning_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        
        # Move version to archive
        version_dir = Path(self.versions[version]['path'])
        archive_path = archive_dir / f"{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(version_dir), str(archive_path))
        
        # Update version tracking
        self.versions[version]['status'] = 'archived'
        self._save_versions()
    
    def restore_version(self, version: str) -> None:
        """Restore an archived version."""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        
        if self.versions[version]['status'] != 'archived':
            raise ValueError(f"Version {version} is not archived")
        
        # Find archived version
        archive_dir = self.versioning_dir / "archive"
        archived_versions = [d for d in archive_dir.iterdir() if d.name.startswith(version)]
        if not archived_versions:
            raise ValueError(f"Archived version {version} not found")
        
        # Restore most recent archived version
        latest_archive = max(archived_versions, key=lambda x: x.stat().st_mtime)
        version_dir = self.versioning_dir / version
        shutil.move(str(latest_archive), str(version_dir))
        
        # Update version tracking
        self.versions[version]['status'] = 'active'
        self._save_versions()
    
    def get_version_history(self, version: str) -> List[Dict[str, Any]]:
        """Get version history from git."""
        repo = git.Repo(str(self.versioning_dir))
        
        # Get commits that modified the version
        version_path = Path(self.versions[version]['path']).relative_to(self.versioning_dir)
        commits = list(repo.iter_commits(paths=str(version_path)))
        
        history = []
        for commit in commits:
            history.append({
                'commit': commit.hexsha,
                'author': commit.author.name,
                'date': datetime.fromtimestamp(commit.committed_date).isoformat(),
                'message': commit.message.strip()
            })
        
        return history
    
    def create_version_branch(self, version: str, branch_name: str) -> None:
        """Create a git branch for a specific version."""
        repo = git.Repo(str(self.versioning_dir))
        
        # Get commit for version
        version_path = Path(self.versions[version]['path']).relative_to(self.versioning_dir)
        commits = list(repo.iter_commits(paths=str(version_path)))
        if not commits:
            raise ValueError(f"No commits found for version {version}")
        
        # Create branch
        repo.create_head(branch_name, commits[0])
    
    def merge_version_branches(self, source_branch: str, target_branch: str) -> None:
        """Merge two version branches."""
        repo = git.Repo(str(self.versioning_dir))
        
        # Checkout target branch
        repo.git.checkout(target_branch)
        
        # Merge source branch
        try:
            repo.git.merge(source_branch)
        except git.exc.GitCommandError as e:
            logger.error(f"Merge failed: {e}")
            repo.git.merge('--abort')
            raise
    
    def get_version_dependencies(self, version: str) -> Dict[str, Any]:
        """Get version dependencies."""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        
        metadata = self.versions[version]['metadata']
        
        return {
            'python_version': metadata.get('python_version', 'unknown'),
            'dependencies': metadata.get('dependencies', {}),
            'requirements': metadata.get('requirements', [])
        }
    
    def validate_version(self, version: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Validate a specific version."""
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