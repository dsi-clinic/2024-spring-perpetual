# Perpetual

### Steps to take:

1. Follow the setup instructions below and ensure that the config.ini file is available in the root directory.
2. Run the file 'GetDistanceMatrix.py' using the command `python GetDistanceMatrix.py <filepath>`. This will generate a distance matrix as a .npy file with a timestamp in the data folder. 
An example command is: `python GetDistanceMatrix.py data/indoor_outdoor.csv` 


Note: I have updated the indoor_outdoor.csv file to include Moody Gardens as the starting location. Moreover, the coordinates have been rounded to 5 decimal places. 

### Setup Instructions

#### Create a Virtual Environment
`python3 -m venv perpetual_env`

#### Activate the Virtual Environment
- On Windows
`perpetual_env\Scripts\activate`

- On Unix or MacOS
`source perpetual_env/bin/activate`

#### Install Dependencies
`pip install -r requirements.txt`
