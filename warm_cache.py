#!/usr/bin/env python3
"""
Cache warmup utility for the Ecological Suitability tab.
Run this to pre-load and cache all data for faster dashboard performance.
"""

import sys
from pathlib import Path

# Add tick directory to path
sys.path.insert(0, str(Path(__file__).parent))

def warm_cache():
    """Pre-load and cache all data."""
    print("🔄 Warming cache for Ecological Suitability tab...")
    print("=" * 50)
    
    try:
        from ui.tabs.suitability import _get_notebook_data, _build_enhanced_map_figure
        from ui.tabs.suitability import DEFAULT_ENHANCED_LAYERS, ENHANCED_LAYER_OPTIONS
        
        # Load and cache all data
        print("Loading notebook data...")
        data = _get_notebook_data()
        
        if not data.available:
            print("❌ Data not available - check that notebook outputs exist")
            return False
        
        print(f"✅ Data loaded successfully")
        print(f"   - Suitability: {len(data.suitability_grid):,} points")
        print(f"   - Occurrence: {len(data.occurrence_points):,} points")
        
        # Pre-generate map figures for common layer combinations
        print("\nPre-generating map figures...")
        
        layer_combinations = [
            DEFAULT_ENHANCED_LAYERS,  # Default view
            ["suitability"],  # Suitability only
            ["occurrence"],  # Occurrence only
            ["suitability", "occurrence"],  # Both main layers
        ]
        
        for i, layers in enumerate(layer_combinations, 1):
            print(f"  {i}/{len(layer_combinations)}: {', '.join(layers)}")
            try:
                figure = _build_enhanced_map_figure(data, layers)
                print(f"     ✅ Generated and cached")
            except Exception as e:
                print(f"     ❌ Error: {e}")
        
        # Pre-load images (done automatically when figures are created)
        print("\nImage caching handled automatically ✅")
        
        print("\n" + "=" * 50)
        print("🚀 Cache warmup completed!")
        print("Dashboard should now load very quickly.")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the tick directory and dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point."""
    success = warm_cache()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()