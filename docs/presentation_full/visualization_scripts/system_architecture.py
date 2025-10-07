#!/usr/bin/env python3
"""
System Architecture Visualization Script
Creates diagrams showing the complete star tracker simulation pipeline architecture.

Generates:
1. Pipeline flowchart showing end-to-end data flow
2. Multi-level architecture diagram (optical → electronic → computational)
3. Data flow diagram showing PSF files → sensor simulation → algorithm processing
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
import numpy as np
import os
from pathlib import Path

# Set up output directory
output_dir = Path(__file__).parent.parent / "figures"
output_dir.mkdir(exist_ok=True)

def create_pipeline_flowchart():
    """Create end-to-end pipeline flowchart with physical processes labeled."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Define colors for different process types
    colors = {
        'optical': '#FF6B6B',    # Red for optical processes
        'electronic': '#4ECDC4', # Teal for electronic processes  
        'computational': '#45B7D1', # Blue for computational processes
        'output': '#96CEB4'      # Green for outputs
    }
    
    # Define box positions and sizes
    box_width = 2.2
    box_height = 0.8
    y_positions = [8, 6.5, 5, 3.5, 2, 0.5]
    
    # Pipeline stages with labels and types
    stages = [
        ("Zemax Optical PSFs\n(128×128, 0.5µm/pixel)", colors['optical']),
        ("CMV4000 Sensor Simulation\n(Quantum efficiency, read noise)", colors['electronic']),
        ("Photon-Level Simulation\n(Poisson statistics, electron conversion)", colors['electronic']),
        ("Star Detection & Centroiding\n(BAST algorithms, sub-pixel accuracy)", colors['computational']),
        ("Bearing Vector Calculation\n(Pixel → angular coordinates)", colors['computational']),
        ("Attitude Determination\n(Triangle matching, QUEST algorithm)", colors['output'])
    ]
    
    # Draw boxes and labels
    boxes = []
    for i, (label, color) in enumerate(stages):
        x_center = 8
        y_center = y_positions[i]
        
        # Create fancy rounded rectangle
        box = FancyBboxPatch(
            (x_center - box_width/2, y_center - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(box)
        boxes.append((x_center, y_center))
        
        # Add text
        ax.text(x_center, y_center, label, ha='center', va='center', 
                fontsize=11, fontweight='bold', wrap=True)
    
    # Draw arrows between stages
    for i in range(len(boxes) - 1):
        x1, y1 = boxes[i]
        x2, y2 = boxes[i + 1]
        
        arrow = FancyArrowPatch(
            (x1, y1 - box_height/2 - 0.1),
            (x2, y2 + box_height/2 + 0.1),
            arrowstyle='->', 
            mutation_scale=20,
            color='black',
            linewidth=3
        )
        ax.add_patch(arrow)
    
    # Add side annotations for physical realism
    annotations = [
        (2, 8, "Real optical\naberrations"),
        (2, 6.5, "Hardware-accurate\nsensor model"),
        (2, 5, "Photon noise\nmodeling"),
        (2, 3.5, "Validated\nalgorithms"),
        (2, 2, "Sub-arcsec\naccuracy"),
        (2, 0.5, "Spacecraft\nattitude")
    ]
    
    for x, y, text in annotations:
        ax.text(x, y, text, ha='center', va='center', fontsize=10, 
                style='italic', color='gray',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.5))
    
    # Add title and formatting
    ax.set_xlim(-1, 17)
    ax.set_ylim(-1, 10)
    ax.set_title('Star Tracker Simulation Pipeline\nComplete Digital Twin Architecture', 
                 fontsize=18, fontweight='bold', pad=20)
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['optical'], label='Optical Processes'),
        patches.Patch(color=colors['electronic'], label='Electronic Processes'),
        patches.Patch(color=colors['computational'], label='Computational Processes'),
        patches.Patch(color=colors['output'], label='Final Output')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pipeline_flowchart.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'pipeline_flowchart.pdf', bbox_inches='tight')
    print(f"Saved pipeline flowchart to {output_dir}")

def create_multilevel_architecture():
    """Create multi-level architecture diagram showing optical → electronic → computational layers."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Define layer positions
    layers = [
        {'name': 'Optical Layer', 'y': 7, 'color': '#FF6B6B', 'components': [
            'Zemax PSF Models', 'Field Angle Effects', 'Aberration Modeling', 'Wavelength Integration'
        ]},
        {'name': 'Electronic Layer', 'y': 5, 'color': '#4ECDC4', 'components': [
            'CMV4000 Sensor', 'Quantum Efficiency', 'Read Noise Model', 'Pixel Response'
        ]},
        {'name': 'Computational Layer', 'y': 3, 'color': '#45B7D1', 'components': [
            'BAST Detection', 'Centroiding Algorithms', 'Triangle Matching', 'QUEST Solver'
        ]},
        {'name': 'Output Layer', 'y': 1, 'color': '#96CEB4', 'components': [
            'Bearing Vectors', 'Attitude Quaternion', 'Accuracy Metrics', 'Performance Data'
        ]}
    ]
    
    # Draw layers
    for layer in layers:
        y = layer['y']
        color = layer['color']
        
        # Main layer box
        layer_box = Rectangle((1, y-0.4), 12, 0.8, 
                             facecolor=color, alpha=0.3, 
                             edgecolor=color, linewidth=3)
        ax.add_patch(layer_box)
        
        # Layer title
        ax.text(0.5, y, layer['name'], ha='right', va='center', 
                fontsize=14, fontweight='bold')
        
        # Component boxes
        comp_width = 2.8
        comp_spacing = 3.0
        start_x = 1.5
        
        for i, component in enumerate(layer['components']):
            x = start_x + i * comp_spacing
            
            comp_box = FancyBboxPatch(
                (x, y-0.3), comp_width, 0.6,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor='black',
                linewidth=1,
                alpha=0.7
            )
            ax.add_patch(comp_box)
            
            ax.text(x + comp_width/2, y, component, ha='center', va='center',
                   fontsize=9, fontweight='bold', wrap=True)
    
    # Draw connections between layers
    for i in range(len(layers) - 1):
        y1 = layers[i]['y'] - 0.4
        y2 = layers[i+1]['y'] + 0.4
        
        for j in range(4):
            x = 2.9 + j * 3.0
            arrow = FancyArrowPatch(
                (x, y1), (x, y2),
                arrowstyle='->', 
                mutation_scale=15,
                color='gray',
                linewidth=2,
                alpha=0.7
            )
            ax.add_patch(arrow)
    
    # Add data flow annotations
    data_flows = [
        (14.5, 6, "PSF Data\n(intensity maps)"),
        (14.5, 4, "Electron Counts\n(with noise)"),
        (14.5, 2, "Pixel Coordinates\n(sub-pixel accuracy)"),
        (14.5, 0, "Attitude Solution\n(quaternion)")
    ]
    
    for x, y, text in data_flows:
        ax.text(x, y, text, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
    
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.5)
    ax.set_title('Multi-Level Architecture\nPhysical Simulation Layers', 
                 fontsize=18, fontweight='bold', pad=20)
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'multilevel_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'multilevel_architecture.pdf', bbox_inches='tight')
    print(f"Saved multi-level architecture to {output_dir}")

def create_data_flow_diagram():
    """Create data flow diagram showing PSF files → sensor simulation → algorithm processing."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Define the data flow stages
    stages = [
        {'name': 'PSF Files\n(Zemax Output)', 'x': 2, 'y': 4, 'width': 2, 'height': 1.5, 'color': '#FF6B6B'},
        {'name': 'PSF Parser\n(Metadata Extraction)', 'x': 5.5, 'y': 4, 'width': 2, 'height': 1.5, 'color': '#FFA500'},
        {'name': 'Sensor Projection\n(CMV4000 Model)', 'x': 9, 'y': 4, 'width': 2, 'height': 1.5, 'color': '#4ECDC4'},
        {'name': 'Photon Simulation\n(Noise Addition)', 'x': 12.5, 'y': 4, 'width': 2, 'height': 1.5, 'color': '#4ECDC4'},
        {'name': 'Detection\n(Threshold & Grouping)', 'x': 4, 'y': 1, 'width': 2, 'height': 1.5, 'color': '#45B7D1'},
        {'name': 'Centroiding\n(Sub-pixel Accuracy)', 'x': 7.5, 'y': 1, 'width': 2, 'height': 1.5, 'color': '#45B7D1'},
        {'name': 'Bearing Vectors\n(Angular Coordinates)', 'x': 11, 'y': 1, 'width': 2, 'height': 1.5, 'color': '#96CEB4'}
    ]
    
    # Draw stages
    for stage in stages:
        # Create rounded rectangle
        box = FancyBboxPatch(
            (stage['x'] - stage['width']/2, stage['y'] - stage['height']/2),
            stage['width'], stage['height'],
            boxstyle="round,pad=0.1",
            facecolor=stage['color'],
            edgecolor='black',
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(box)
        
        # Add text
        ax.text(stage['x'], stage['y'], stage['name'], ha='center', va='center',
                fontsize=10, fontweight='bold')
    
    # Define arrows with data descriptions
    arrows = [
        # Top row
        {'from': (3, 4), 'to': (4.5, 4), 'label': 'Raw PSF\nData'},
        {'from': (6.5, 4), 'to': (8, 4), 'label': 'Parsed\nIntensity Map'},
        {'from': (10, 4), 'to': (11.5, 4), 'label': 'Projected\nPixel Array'},
        # Down to bottom row
        {'from': (12.5, 3.25), 'to': (5, 2.25), 'label': 'Noisy\nDetector Image', 'curved': True},
        # Bottom row
        {'from': (5, 1), 'to': (6.5, 1), 'label': 'Detected\nPixel Groups'},
        {'from': (8.5, 1), 'to': (10, 1), 'label': 'Centroid\nCoordinates'}
    ]
    
    # Draw arrows
    for arrow in arrows:
        if arrow.get('curved', False):
            # Curved arrow for long connection
            arrow_patch = FancyArrowPatch(
                arrow['from'], arrow['to'],
                arrowstyle='->', 
                mutation_scale=20,
                color='black',
                linewidth=2,
                connectionstyle="arc3,rad=0.3"
            )
        else:
            # Straight arrow
            arrow_patch = FancyArrowPatch(
                arrow['from'], arrow['to'],
                arrowstyle='->', 
                mutation_scale=20,
                color='black',
                linewidth=2
            )
        
        ax.add_patch(arrow_patch)
        
        # Add label
        mid_x = (arrow['from'][0] + arrow['to'][0]) / 2
        mid_y = (arrow['from'][1] + arrow['to'][1]) / 2
        if arrow.get('curved', False):
            mid_y += 0.5  # Offset for curved arrow
        
        ax.text(mid_x, mid_y + 0.3, arrow['label'], ha='center', va='center',
                fontsize=9, style='italic',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Add file format annotations
    file_formats = [
        (2, 2.5, "Gen_1/\n128×128\n0.5µm/pixel"),
        (2, 5.5, "Gen_2/\n32×32\n0.232µm/pixel"),
        (14, 2.5, "5.5µm pixel\n2048×2048\nCMV4000")
    ]
    
    for x, y, text in file_formats:
        ax.text(x, y, text, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.7))
    
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6.5)
    ax.set_title('Data Flow Architecture\nPSF Processing Pipeline', 
                 fontsize=18, fontweight='bold', pad=20)
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'data_flow_diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'data_flow_diagram.pdf', bbox_inches='tight')
    print(f"Saved data flow diagram to {output_dir}")

def main():
    """Generate all system architecture visualizations."""
    print("Generating system architecture visualizations...")
    
    create_pipeline_flowchart()
    create_multilevel_architecture()
    create_data_flow_diagram()
    
    print(f"\nAll architecture diagrams saved to: {output_dir}")
    print("Generated files:")
    print("- pipeline_flowchart.png/pdf")
    print("- multilevel_architecture.png/pdf")  
    print("- data_flow_diagram.png/pdf")

if __name__ == "__main__":
    main()