# ML Optimization of Shielding for Molten Salt Microreactors

## Introduction
This project focuses on optimizing the shielding of molten salt microreactors using machine learning. It aims to reduce the mass and cost of reactor shields while maintaining safety standards. Our approach combines predictive machine learning models with the Gekko Optimization Suite for efficient shield design.

## Repository Contents
- `shielding10.py`: The main Python script implementing the optimization algorithm.
- `data.pkl`: Data sets used for machine learning model training.
- `maxVals.csv`: Maximum values

## Background
Radiation shielding is crucial for nuclear reactors. Traditional shields can be bulky and expensive, limiting the application in small, modular, mobile reactors. Our work employs machine learning to optimize shield material and configuration, significantly reducing computational time and resources.

## Methods
- **Shield Geometry & Simulation**: Utilizes a 1-D radial model for shield geometry.
- **Machine Learning Model**: A predictive model employing a Multilayer Perceptron to estimate shielding effectiveness.
- **Optimization Algorithm**: Implemented using the Gekko Optimization Suite to optimize shield materials based on the ML model.

## Results
Our approach achieved a significant reduction in shield mass (10.8%) and cost (11.9%) compared to traditional methods, while maintaining safety standards.

## How to Run the Code
1. Ensure Python 3.x and necessary libraries (listed in `requirements.txt`) are installed.
2. Run `python shielding10.py` to execute the optimization algorithm.
3. Output will include optimized shield configurations and performance metrics.

## Dependencies
- Python
- Gekko Optimization Suite
- Sci-Kit Learn
- Additional dependencies are listed in `requirements.txt`.

## Citing Our Work
If you use our code or refer to our research, please cite:

Larsen, A., Lee, R., Wilson, C., Hedengren, J.D., Benson, J., Memmott, M., Multi-Objective Optimization of Molten Salt Microreactor Shielding Employing Machine Learning, Preprint submitted for publication.

## Contact
For queries or collaborations, contact the corresponding author:
- Dr. Matthew Memmott
- Email: memmott@byu.edu

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
Special thanks to Alphatech Research Corp. for funding and support, and all contributors to the project.
