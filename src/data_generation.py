# src/data_generation.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import cv2
from PIL import Image
from fpdf import FPDF

def generate_all_data():
    # ... (all code from the answer, with image saving to data/images/)
    # Save CSV files to data/raw/
    df_prod.to_csv('data/raw/production.csv', index=False)
    df_sensor.to_csv('data/raw/sensors.csv', index=False)
    df_images.to_csv('data/raw/image_metadata.csv', index=False)
    df_text.to_csv('data/raw/maintenance_notes.csv', index=False)
    # PDFs saved to data/raw/

if __name__ == "__main__":
    generate_all_data()