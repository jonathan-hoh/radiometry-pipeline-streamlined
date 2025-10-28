#!/usr/bin/env python3
"""
run_gui.py - Star Tracker GUI Application Launcher

Launch script for the Star Tracker desktop GUI application.
Provides the main entry point for the PyQt6-based interface.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set PYTHONPATH environment variable
os.environ['PYTHONPATH'] = str(project_root)

def check_dependencies():
    """Check if required dependencies are installed."""
    missing_deps = []
    
    try:
        import PyQt6
    except ImportError:
        missing_deps.append("PyQt6")
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")
        
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
        
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nInstall missing dependencies with:")
        print(f"   pip install {' '.join(missing_deps)}")
        return False
        
    return True

def main():
    """Main application launcher."""
    print("Starting Star Tracker GUI...")
    
    # Check dependencies
    if not check_dependencies():
        return 1
        
    try:
        # Import and run the GUI application
        from GUI.main_window import main as gui_main
        return gui_main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nTry running with PYTHONPATH set:")
        print("   PYTHONPATH=. python run_gui.py")
        return 1
        
    except Exception as e:
        print(f"❌ Application error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())