# Monocular Depth Estimation

This repository contains the monocular depth estimation part of a larger robotics and computer vision capstone project. The current work focuses on estimating depth from single RGB images using **Depth Anything V2 (DAv2)**, along with supporting experiments involving **YOLO** for object understanding in related vision tasks.

At this stage, the monocular depth estimation work has been tested on **sample images captured and processed on a computer**, not yet on a live ESP32/Hiwonder camera stream.

## Repository Structure

```text
Monocular_dept_estimation/
├── depth_output/
├── src/
│   └── monocular_depth_images.py
├── test/
├── LICENSE
└── sort.py
```

### Folder Details

- `src/monocular_depth_images.py` — main source file for running monocular depth estimation on images.
- `test/` — test scripts, experiments, or evaluation files.
- `depth_output/` — saved output depth maps and generated visual results.
- `sort.py` — helper script used in the project workflow, if applicable.
- `LICENSE` — repository license information.

## Features

- Monocular depth estimation from a single RGB image using DAv2-related workflow.
- Local image-based testing and output generation.
- Support for visual experiments connected to YOLO-based perception tasks.
- Organized project structure for further robotics integration.

## Requirements

This project may use the following Python packages based on the current toolchain involving **YOLO** and **Depth Anything V2**:

```txt
numpy
opencv-python
pillow
matplotlib
torch
torchvision
timm
ultralytics
transformers
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

## How to Run

From the project root, run the main depth estimation script:

```bash
python src/monocular_depth_images.py
```

If your workflow uses test images from a folder or command-line arguments, update the paths in the script before running.

## Output

Generated depth maps, processed images, or visualizations are stored in:

```text
depth_output/
```

These outputs can be used for qualitative inspection and future integration with navigation or scene understanding modules.

## Relation to Broader Project

This repository supports a larger assistive robotics vision pipeline that includes:

1. Area Inspection — detecting objects such as people and chairs.
2. Pattern recognition and action — identifying crosswalk paths for safe crossing.
3. Follow — recognizing and tracking a person while maintaining distance.
4. Landmark saving and navigation — storing and later matching landmarks such as doors.
5. Descriptor — generating short scene descriptions such as “A chair ahead 2 metres”.

Depth estimation is particularly relevant for obstacle awareness, spatial understanding, and future navigation-related features.


## Future Improvements


- Live monocular depth estimation using camera feeds.
- Integration of depth with object detection for richer scene understanding.
- Better depth scaling, calibration, and visualization.
- Integration into navigation and landmark-based robotic behaviors.

## License

This project includes a `LICENSE` file in the repository root.
