import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from math import radians
import ezdxf
from io import BytesIO, StringIO
import pytesseract
from pdf2image import convert_from_path
import re
import tempfile
import os
from openai import OpenAI
import json

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def format_bearing_concise(bearing_desc):
    """Convert verbose bearing description to concise surveyor's notation."""
    try:
        # Extract bearing components from the original text
        pattern = r'(North|South)\s+(\d+)\s*(?:°|degrees?|deg|\s)\s*(\d+)\s*(?:\'|′|minutes?|min|\s)\s*(?:(\d+)\s*(?:"|″|seconds?|sec|\s)\s+)?(East|West)'
        match = re.search(pattern, bearing_desc, re.IGNORECASE)

        if match:
            cardinal_ns, deg, min, sec, cardinal_ew = match.groups()
            # Format to concise notation
            ns = 'N' if cardinal_ns.lower() == 'north' else 'S'
            ew = 'E' if cardinal_ew.lower() == 'east' else 'W'
            sec = f" {int(sec):02d}s" if sec else ""
            return f"{ns} {deg}d {min}m{sec} {ew}"
    except Exception:
        return bearing_desc  # Return original if parsing fails
    return bearing_desc

def extract_bearings_with_gpt(text):
    """Use GPT to extract bearings from text."""
    try:
        # Create a prompt that instructs GPT to find bearings
        prompt = """Extract all bearings, distances, and monuments from the following legal description text. 
        Format each bearing exactly like this example, one per line:
        BEARING: North 45 degrees 30 minutes East DISTANCE: 100.00 feet MONUMENT: to an iron pin
        Note: Seconds are optional and should be omitted if not present in the text.
        The monument is any description that appears after 'feet' and before 'thence' or 'running thence'.
        If no monument is mentioned, leave it blank after MONUMENT:

        Text to analyze:
        """ + text

        # Call GPT-4 with the prompt
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0
        )

        # Get the response text
        result_text = response.choices[0].message.content

        # Parse the response into our bearing format
        bearings = []
        for line in result_text.split('\n'):
            if line.strip().startswith('BEARING:'):
                try:
                    # Split into bearing, distance, and monument parts
                    parts = line.split('DISTANCE:')
                    if len(parts) != 2:
                        continue

                    bearing_text = parts[0].replace('BEARING:', '').strip()
                    distance_monument_text = parts[1].strip()

                    # Split distance and monument
                    distance_parts = distance_monument_text.split('MONUMENT:', 1)
                    distance_text = distance_parts[0].strip()
                    monument_text = distance_parts[1].strip() if len(distance_parts) > 1 else ""

                    # Parse bearing components - make seconds optional
                    pattern = r'(North|South)\s+(\d+)\s*(?:°|degrees?|deg|\s)\s*(\d+)\s*(?:\'|′|minutes?|min|\s)\s*(?:(\d+)\s*(?:"|″|seconds?|sec|\s)\s+)?(East|West)'
                    match = re.search(pattern, bearing_text, re.IGNORECASE)

                    if match:
                        cardinal_ns, deg, min, sec, cardinal_ew = match.groups()
                        # Default seconds to 0 if not found
                        sec = int(sec) if sec else 0

                        # Parse distance
                        distance_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:feet|ft|\')', distance_text, re.IGNORECASE)
                        distance = float(distance_match.group(1)) if distance_match else 0.00

                        bearing = {
                            'cardinal_ns': cardinal_ns,
                            'degrees': int(deg),
                            'minutes': int(min),
                            'seconds': sec,
                            'cardinal_ew': cardinal_ew,
                            'distance': distance,
                            'monument': monument_text,
                            'original_text': line.strip()
                        }
                        bearings.append(bearing)
                    else:
                        st.warning(f"Could not match bearing pattern in: {bearing_text}")
                except Exception as parse_error:
                    st.warning(f"Could not parse line: {line}\nError: {str(parse_error)}")
                    continue

        return bearings
    except Exception as e:
        st.error(f"Error using GPT to parse text: {str(e)}")
        return []

def extract_bearings_from_text(text):
    """Extract bearings from text with more flexible pattern matching."""
    try:
        # Find the first occurrence of North or South with more flexible pattern
        first_bearing_pattern = r'(North|South)\s*\d+(?:\s*°|\s*degrees|\s+)'
        start_match = re.search(first_bearing_pattern, text, re.IGNORECASE)

        if not start_match:
            st.warning("Could not find any bearings in the text")
            st.info("Looking for text starting with 'North' or 'South' followed by numbers")
            return []

        # Extract text from the first bearing onwards
        relevant_text = text[start_match.start():]

        # Pattern matches formats like "North 11° 22' 33" East" with flexible separators
        pattern = r'(North|South)\s*(\d+)\s*(?:°|degrees|deg|\s)\s*(\d+)\s*(?:\'|′|minutes|min|\s)\s*(\d+)\s*(?:"|″|seconds|sec|\s)\s*(East|West)'

        # Split text into potential segments at common delimiters
        segments = re.split(r'(?:thence|;|,|\n)', relevant_text)

        bearings = []
        for segment in segments:
            # Look for bearing pattern in each segment
            match = re.search(pattern, segment, re.IGNORECASE)
            if match:
                cardinal_ns, deg, min, sec, cardinal_ew = match.groups()

                # Look for distance and monument in the same segment
                distance = 0.00  # Default distance
                monument = ""    # Default monument

                # More flexible distance pattern
                distance_pattern = r'(\d+[.,\d]*)\s*(?:feet|ft|\')'
                distance_match = re.search(distance_pattern, segment, re.IGNORECASE)

                if distance_match:
                    # Remove all punctuation and add decimal point for 2 decimal places
                    distance_str = re.sub(r'[.,]', '', distance_match.group(1))
                    distance = float(distance_str) / 100  # Convert to decimal form

                    # Look for monument after "feet" and before "thence"
                    monument_match = re.search(r'(?:feet|ft|\')\s*(.*?)(?:\s+(?:running\s+)?thence|$)', segment[distance_match.end():], re.IGNORECASE)
                    if monument_match:
                        monument = monument_match.group(1).strip()

                bearings.append({
                    'cardinal_ns': 'North' if cardinal_ns.lower() == 'north' else 'South',
                    'degrees': int(deg),
                    'minutes': int(min),
                    'seconds': int(sec),
                    'cardinal_ew': 'East' if cardinal_ew.lower() == 'east' else 'West',
                    'distance': distance,
                    'monument': monument,
                    'original_text': segment.strip()
                })

        return bearings
    except Exception as e:
        st.error(f"Error parsing text: {str(e)}")
        return []

def dms_to_decimal(degrees, minutes, seconds, cardinal_ns, cardinal_ew):
    """Convert DMS (Degrees, Minutes, Seconds) to decimal degrees."""
    decimal = float(degrees) + float(minutes)/60 + float(seconds)/3600

    # Adjust based on cardinal directions
    if cardinal_ns == 'South':
        decimal = -decimal

    # Convert to bearing (clockwise from north)
    if cardinal_ew == 'East':
        decimal = 90 - decimal
    else:  # West
        decimal = 270 + decimal

    return decimal % 360

def decimal_to_dms(decimal_degrees):
    """Convert decimal degrees back to DMS format."""
    # Convert bearing to compass reading
    if 0 <= decimal_degrees <= 90:
        cardinal_ns = 'North'
        cardinal_ew = 'East'
        angle = 90 - decimal_degrees
    elif 90 < decimal_degrees <= 180:
        cardinal_ns = 'South'
        cardinal_ew = 'East'
        angle = decimal_degrees - 90
    elif 180 < decimal_degrees <= 270:
        cardinal_ns = 'South'
        cardinal_ew = 'West'
        angle = 270 - decimal_degrees
    else:
        cardinal_ns = 'North'
        cardinal_ew = 'West'
        angle = decimal_degrees - 270

    degrees = int(angle)
    minutes_float = (angle - degrees) * 60
    minutes = int(minutes_float)
    seconds = int((minutes_float - minutes) * 60)

    return cardinal_ns, degrees, minutes, seconds, cardinal_ew

def calculate_endpoint(start_point, bearing, distance):
    """Calculate endpoint coordinates given start point, bearing and distance."""
    bearing_rad = radians(bearing)
    dx = distance * np.sin(bearing_rad)
    dy = distance * np.cos(bearing_rad)
    end_x = start_point[0] + dx
    end_y = start_point[1] + dy
    return [end_x, end_y]

def create_dxf():
    """Create a DXF file from the current lines."""
    if st.session_state.lines.empty:
        st.error("No lines to export")
        return None

    try:
        # Create new document with setup=True
        doc = ezdxf.new(setup=True)
        msp = doc.modelspace()

        # Create and configure dimension style
        dim_style = doc.dimstyles.new('CUSTOM_STYLE')
        dim_style.dxf.dimtxt = 0.8  # Text height
        dim_style.dxf.dimgap = 0.1  # Gap between text and dimension line
        dim_style.dxf.dimasz = 0.4  # Arrow size
        dim_style.dxf.dimclrd = 1   # Dimension line color (1=red)
        dim_style.dxf.dimclrt = 2   # Dimension text color (2=yellow)

        # Add POB text and arrow
        try:
            # Add POB text
            msp.add_text(
                "POB",
                dxfattribs={
                    "layer": "Text",
                    "height": 1.0,
                    "insert": (3, -3)  # Offset from origin
                }
            )
            # Add arrow to POB
            msp.add_line(
                (3, -3),  # Start at text location
                (0, 0),   # End at origin
                dxfattribs={"layer": "POB_Arrow"}
            )
            # Add circle at POB (origin)
            msp.add_circle((0, 0), radius=0.5, dxfattribs={"layer": "Points"})
        except Exception as pob_error:
            st.warning(f"Error adding POB annotation: {str(pob_error)}")

        # Add each line
        for idx, row in st.session_state.lines.iterrows():
            start = (float(row['start_x']), float(row['start_y']))
            end = (float(row['end_x']), float(row['end_y']))

            try:
                # Add the line
                msp.add_line((start[0], start[1]), (end[0], end[1]), dxfattribs={"layer": "Lines"})

                # Add circles at start and end points
                msp.add_circle((start[0], start[1]), radius=0.5, dxfattribs={"layer": "Points"})
                msp.add_circle((end[0], end[1]), radius=0.5, dxfattribs={"layer": "Points"})

                # Calculate angle and offset for dimension line
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                angle = np.arctan2(dy, dx)
                # Offset perpendicular to the line
                offset = 2.0  # Distance to offset dimension line
                offset_x = -np.sin(angle) * offset
                offset_y = np.cos(angle) * offset

                # Add linear dimension
                dim = msp.add_linear_dim(
                    base=(start[0] + offset_x, start[1] + offset_y),  # Base point (offset from line)
                    p1=start,    # Start point
                    p2=end,      # End point
                    angle=0,  # Angle in degrees (horizontal)
                    text=f"{row['distance']:.2f}'",  # Override dimension text
                    dimstyle='CUSTOM_STYLE',
                    override={
                        'dimtad': 1,  # Text position above dimension line
                        'dimtix': 1,  # Force text inside extension lines
                    }
                )

                # Add monument text if available (offset slightly from the points)
                if idx > 0:  # For all points except POB
                    # Add monument text at start point (from previous line's end)
                    prev_row = st.session_state.lines.iloc[idx-1]
                    if 'monument' in prev_row and prev_row['monument']:
                        msp.add_text(
                            prev_row['monument'],
                            dxfattribs={
                                "layer": "Monuments",
                                "height": 0.8,
                                "insert": (start[0] + 1, start[1] + 1)  # Offset text position
                            }
                        )

                # Add monument text at end point of current line
                if 'monument' in row and row['monument']:
                    msp.add_text(
                        row['monument'],
                        dxfattribs={
                            "layer": "Monuments",
                            "height": 0.8,
                            "insert": (end[0] + 1, end[1] + 1)  # Offset text position
                        }
                    )
            except Exception as line_error:
                st.error(f"Error adding line {idx+1}: {str(line_error)}")
                return None

        # Save the file
        filename = "line_drawing.dxf"
        doc.saveas(filename)
        st.write(f"DXF file saved as: {filename}")

        # Read the file back for download
        with open(filename, 'rb') as f:
            return f.read()

    except Exception as e:
        st.error(f"DXF creation error: {str(e)}")
        return None

def create_test_dxf():
    """Create a test DXF file with simple content."""
    try:
        st.write("Creating test DXF file...")
        # Create new document with setup=True
        doc = ezdxf.new(setup=True)
        msp = doc.modelspace()

        # Add test line with only X,Y coordinates
        start_x, start_y = 0, 0
        end_x, end_y = 10, 10
        st.write(f"Adding test line: ({start_x}, {start_y}) to ({end_x}, {end_y})")

        try:
            msp.add_line((start_x, start_y), (end_x, end_y), dxfattribs={"layer": "TestLayer"})
        except Exception as line_error:
            st.error(f"Error adding test line: {str(line_error)}")
            return None

        # Save the file
        filename = "test.dxf"
        doc.saveas(filename)
        st.write(f"Test DXF file saved as: {filename}")

        # Read the file back for download
        with open(filename, 'rb') as f:
            return f.read()

    except Exception as e:
        st.error(f"Test DXF creation error: {str(e)}")
        return None

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'lines' not in st.session_state:
        st.session_state.lines = pd.DataFrame(columns=['start_x', 'start_y', 'end_x', 'end_y', 'bearing', 'distance', 'bearing_desc', 'monument'])
    if 'current_point' not in st.session_state:
        st.session_state.current_point = [0, 0]
    if 'gpt_analysis' not in st.session_state:
        st.session_state.gpt_analysis = None
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = None
    if 'parsed_bearings' not in st.session_state:
        st.session_state.parsed_bearings = None
    if 'pdf_image' not in st.session_state:
        st.session_state.pdf_image = None
    if 'supplemental_info' not in st.session_state:
        st.session_state.supplemental_info = None

def draw_lines():
    """Create a Plotly figure with all lines."""
    fig = go.Figure()

    # Add POB annotation at origin
    fig.add_annotation(
        x=0,
        y=0,
        text="POB",
        showarrow=True,
        arrowhead=2,
        ax=30,  # Offset x position for text
        ay=-30,  # Offset y position for text
        font=dict(size=14),
        arrowsize=1.5,
        arrowwidth=2
    )

    # Draw all lines
    for idx, row in st.session_state.lines.iterrows():
        # Calculate midpoint for text position
        mid_x = (row['start_x'] + row['end_x']) / 2
        mid_y = (row['start_y'] + row['end_y']) / 2

        # Add line
        fig.add_trace(go.Scatter(
            x=[row['start_x'], row['end_x']],
            y=[row['start_y'], row['end_y']],
            mode='lines',
            name=f'Line {idx+1}',
            line=dict(width=2)
        ))

        # Add text label with concise bearing format
        bearing_text = format_bearing_concise(row["bearing_desc"])
        fig.add_trace(go.Scatter(
            x=[mid_x],
            y=[mid_y],
            mode='text',
            text=[f"{bearing_text}<br>{row['distance']:.2f} ft"],
            textposition='top center',
            hoverinfo='text',
            showlegend=False
        ))

        # Add points
        fig.add_trace(go.Scatter(
            x=[row['start_x']],
            y=[row['start_y']],
            mode='markers',
            name=f'Point {idx}',
            marker=dict(size=8)
        ))

    # Add final point
    if not st.session_state.lines.empty:
        fig.add_trace(go.Scatter(
            x=[st.session_state.lines.iloc[-1]['end_x']],
            y=[st.session_state.lines.iloc[-1]['end_y']],
            mode='markers',
            name=f'Point {len(st.session_state.lines)}',
            marker=dict(size=8)
        ))

    # Update layout with scroll/pan enabled and zoom disabled
    fig.update_layout(
        showlegend=False,
        title='Line Drawing',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
            scaleanchor="y",
            scaleratio=1,
            constrain="domain"
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
            constrain="domain"
        ),
        width=800,
        height=600,
        dragmode='pan'  # Enable panning by default
    )

    # Configure interaction modes
    fig.update_layout(
        modebar=dict(
            remove=['zoomIn', 'zoomOut', 'autoScale'],
            add=['pan']
        )
    )

    return fig

def create_rectangle(start_point, side_length):
    """Create a rectangle-like shape with slightly randomized angles that closes back to start."""
    lines = []
    current = start_point
    first_point = start_point

    # Define three slightly randomized directions
    directions = [
        ("North", np.random.randint(0, 15), np.random.randint(0, 60), np.random.randint(0, 60), "East"),    # ~North
        ("North", np.random.randint(75, 90), np.random.randint(0, 60), np.random.randint(0, 60), "East"),   # ~East
        ("South", np.random.randint(0, 15), np.random.randint(0, 60), np.random.randint(0, 60), "East"),    # ~South
    ]

    # Draw first three lines with random angles
    for cardinal_ns, deg, min, sec, cardinal_ew in directions:
        # Convert DMS to decimal
        bearing = dms_to_decimal(deg, min, sec, cardinal_ns, cardinal_ew)

        # Calculate endpoint
        end_point = calculate_endpoint(current, bearing, side_length)

        # Create bearing description
        bearing_desc = f"{cardinal_ns} {deg}° {min}' {sec}\" {cardinal_ew}"

        # Add line
        lines.append({
            'start_x': current[0],
            'start_y': current[1],
            'end_x': end_point[0],
            'end_y': end_point[1],
            'bearing': bearing,
            'bearing_desc': bearing_desc,
            'distance': side_length
        })

        current = end_point

    # Calculate the bearing and distance for the closing line
    dx = first_point[0] - current[0]
    dy = first_point[1] - current[1]
    closing_distance = np.sqrt(dx*dx + dy*dy)
    closing_bearing = np.degrees(np.arctan2(dx, dy)) % 360

    # Convert the closing bearing to DMS format
    cardinal_ns, deg, min, sec, cardinal_ew = decimal_to_dms(closing_bearing)
    bearing_desc = f"{cardinal_ns} {deg}° {min}' {sec}\" {cardinal_ew}"

    # Add the closing line
    lines.append({
        'start_x': current[0],
        'start_y': current[1],
        'end_x': first_point[0],
        'end_y': first_point[1],
        'bearing': closing_bearing,
        'bearing_desc': bearing_desc,
        'distance': closing_distance
    })

    return pd.DataFrame(lines)

def extract_supplemental_info_with_gpt(text):
    """Use GPT to extract Land Lot #, District, and County information."""
    try:
        prompt = """Extract the Land Lot number, District, and County information from the following text.
        Format the response exactly like this example:
        Land Lot: 123
        District: 2nd
        County: Fulton

        Text to analyze:
        """ + text

        # Call GPT-4 with the prompt
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0
        )

        # Get the response text
        result_text = response.choices[0].message.content

        # Parse the response
        land_lot = None
        district = None
        county = None

        for line in result_text.split('\n'):
            if line.strip().startswith('Land Lot:'):
                land_lot = line.replace('Land Lot:', '').strip()
            elif line.strip().startswith('District:'):
                district = line.replace('District:', '').strip()
            elif line.strip().startswith('County:'):
                county = line.replace('County:', '').strip()

        return {'land_lot': land_lot, 'district': district, 'county': county}
    except Exception as e:
        st.error(f"Error extracting supplemental info: {str(e)}")
        return None

def draw_lines_from_bearings():
    """Draw lines using the parsed bearings from session state."""
    if not st.session_state.parsed_bearings:
        return

    for line_num, bearing in enumerate(st.session_state.parsed_bearings):
        # Only process lines with non-zero distance
        distance = bearing['distance']
        if distance > 0:
            # Convert DMS to decimal degrees
            bearing_decimal = dms_to_decimal(
                bearing['degrees'],
                bearing['minutes'],
                bearing['seconds'],
                bearing['cardinal_ns'],
                bearing['cardinal_ew']
            )

            # Calculate new endpoint
            end_point = calculate_endpoint(st.session_state.current_point, bearing_decimal, distance)

            # Create bearing description
            bearing_desc = bearing['original_text']

            # Add new line to DataFrame
            new_line = pd.DataFrame({
                'start_x': [st.session_state.current_point[0]],
                'start_y': [st.session_state.current_point[1]],
                'end_x': [end_point[0]],
                'end_y': [end_point[1]],
                'bearing': [bearing_decimal],
                'bearing_desc': [bearing_desc],
                'distance': [distance],
                'monument': [bearing['monument']] #added monument
            })
            st.session_state.lines = pd.concat([st.session_state.lines, new_line], ignore_index=True)

            # Update current point
            st.session_state.current_point = end_point

def process_pdf(uploaded_file):
    """Process uploaded PDF file and extract bearings."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        # Convert PDF to images
        images = convert_from_path(pdf_path)

        # Store the first page image in session state
        if images:
            # Convert PIL image to bytes for display
            img_byte_arr = BytesIO()
            images[0].save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            st.session_state.pdf_image = img_byte_arr

        # Extract text from each page
        extracted_text = ""
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            extracted_text += f"\n--- Page {i+1} ---\n{text}\n"

        # Clean up temporary file
        os.unlink(pdf_path)

        # Store extracted text in session state
        st.session_state.extracted_text = extracted_text

        # Extract supplemental information first
        if os.environ.get("OPENAI_API_KEY"):
            st.info("Extracting property information...")
            try:
                supplemental_info = extract_supplemental_info_with_gpt(extracted_text)
                if supplemental_info:
                    st.session_state.supplemental_info = supplemental_info
                    st.success("Successfully extracted property information")
            except Exception as e:
                st.error(f"Error extracting property information: {str(e)}")

        # First try GPT extraction for bearings
        bearings = []
        if os.environ.get("OPENAI_API_KEY"):
            st.info("Using GPT to analyze the text for bearings...")
            try:
                bearings = extract_bearings_with_gpt(extracted_text)
                if bearings:
                    st.success(f"Successfully extracted {len(bearings)} bearings using GPT")
                    # Store bearings in session state
                    st.session_state.parsed_bearings = bearings
                    # Automatically draw lines
                    st.session_state.current_point = [0, 0]  # Reset starting point
                    st.session_state.lines = pd.DataFrame(columns=['start_x', 'start_y', 'end_x', 'end_y', 'bearing', 'bearing_desc', 'distance', 'monument'])
                    draw_lines_from_bearings()
                    return bearings
                else:
                    st.warning("GPT analysis found no bearings, falling back to pattern matching...")
            except Exception as e:
                st.error(f"GPT analysis failed: {str(e)}, falling back to pattern matching...")
        else:
            st.warning("No OpenAI API key found, using pattern matching...")

        # Only fall back to pattern matching if GPT failed or found nothing
        st.info("Using pattern matching method...")
        bearings = extract_bearings_from_text(extracted_text)
        if bearings:
            st.success(f"Found {len(bearings)} bearings using pattern matching")
            # Store bearings in session state
            st.session_state.parsed_bearings = bearings
            # Automatically draw lines
            st.session_state.current_point = [0, 0]  # Reset starting point
            st.session_state.lines = pd.DataFrame(columns=['start_x', 'start_y', 'end_x', 'end_y', 'bearing', 'bearing_desc', 'distance', 'monument'])
            draw_lines_from_bearings()
        else:
            st.warning("No bearings found with pattern matching")

        return bearings
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return []

def main():
    # Configure Streamlit for file uploads
    st.set_page_config(
        page_title="Line Drawing Application",
        initial_sidebar_state="expanded",
        layout="wide"
    )

    # Add CORS headers
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type'
    }
    for key, value in headers.items():
        st.markdown(f'<style>header {{-webkit-{key}: {value}; {key}: {value};}}</style>', unsafe_allow_html=True)

    st.title("Line Drawing Application")
    initialize_session_state()

    # Create two columns for the main layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # PDF Upload Section
        st.subheader("Import PDF with Bearings")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            if st.button("Process PDF"):
                bearings = process_pdf(uploaded_file)
                if bearings:
                    st.success(f"Found {len(bearings)} bearings in the PDF")
                    # Initialize session state for all form fields
                    for i in range(4):
                        if i < len(bearings):
                            bearing = bearings[i]
                            st.session_state[f"cardinal_ns_{i}"] = bearing['cardinal_ns']
                            st.session_state[f"degrees_{i}"] = int(bearing['degrees'])
                            st.session_state[f"minutes_{i}"] = int(bearing['minutes'])
                            st.session_state[f"seconds_{i}"] = int(bearing['seconds'])
                            st.session_state[f"cardinal_ew_{i}"] = bearing['cardinal_ew']
                            st.session_state[f"distance_{i}"] = float(bearing['distance'])
                            st.session_state[f"monument_{i}"] = bearing['monument'] #added monument
                        else:
                            # Initialize remaining fields to defaults
                            st.session_state[f"cardinal_ns_{i}"] = "North"
                            st.session_state[f"degrees_{i}"] = 0
                            st.session_state[f"minutes_{i}"] =0
                            st.session_state[f"seconds_{i}"] = 0
                            st.session_state[f"cardinal_ew_{i}"] = "East"
                            st.session_state[f"distance_{i}"] = 0.00
                            st.session_state[f"monument_{i}"] = "" #added monument


    # Show extracted text and analysis in the second column if available
    with col2:
        if st.session_state.extracted_text:
            st.subheader("Extracted Text")
            st.text_area("Raw Text", st.session_state.extracted_text, height=200)

        if st.session_state.parsed_bearings:
            st.subheader("Parsed Bearings")
            for i, bearing in enumerate(st.session_state.parsed_bearings):
                st.text(f"Line {i+1}: {bearing['original_text']}")

    # Line Drawing Section
    st.subheader("Draw Lines")

    # Create a container for all line inputs
    with st.container():
        for line_num in range(4):
            st.write(f"Line {line_num + 1}")
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

            with col1:
                cardinal_ns = st.selectbox(
                    "N/S",
                    ["North", "South"],
                    key=f"cardinal_ns_{line_num}"
                )

            with col2:
                degrees = st.number_input(
                    "Deg",
                    min_value=0,
                    max_value=90,
                    value=st.session_state.get(f"degrees_{line_num}", 0),
                    step=1,
                    format="%d",
                    key=f"degrees_{line_num}"
                )

            with col3:
                minutes = st.number_input(
                    "Min",
                    min_value=0,
                    max_value=59,
                    value=st.session_state.get(f"minutes_{line_num}", 0),
                    format="%d",
                    key=f"minutes_{line_num}"
                )

            with col4:
                seconds = st.number_input(
                    "Sec",
                    min_value=0,
                    max_value=59,
                    value=st.session_state.get(f"seconds_{line_num}", 0),
                    format="%d",
                    key=f"seconds_{line_num}"
                )

            with col5:
                cardinal_ew = st.selectbox(
                    "E/W",
                    ["East", "West"],
                    key=f"cardinal_ew_{line_num}"
                )

            with col6:
                distance = st.number_input(
                    "Distance",
                    min_value=0.0,
                    value=st.session_state.get(f"distance_{line_num}", 0.00),
                    format="%.2f",  # Changed to always show 2 decimal places
                    key=f"distance_{line_num}"
                )
            with col7:
                monument = st.text_input(
                    "Monument",
                    value=st.session_state.get(f"monument_{line_num}", ""),
                    key=f"monument_{line_num}"
                )

    # Control Buttons
    col1, col2, col3, col4, col5 = st.columns(5)

    # Draw Lines button
    with col1:
        if st.button("Draw Lines", use_container_width=True):
            # Reset starting point and lines
            st.session_state.current_point = [0, 0]
            st.session_state.lines = pd.DataFrame(columns=['start_x', 'start_y', 'end_x', 'end_y', 'bearing', 'bearing_desc', 'distance', 'monument'])
            draw_lines_from_bearings()

    # Show Land Lot button
    with col2:
        if st.button("Show Land Lot", use_container_width=True):
            if st.session_state.extracted_text and os.environ.get("OPENAI_API_KEY"):
                st.session_state.supplemental_info = extract_supplemental_info_with_gpt(st.session_state.extracted_text)
                if st.session_state.supplemental_info:
                    st.success("Successfully extracted supplemental information")
            else:
                st.warning("Please process a PDF file first")

    # Clear all button
    with col3:
        if st.button("Clear All", use_container_width=True):
            st.session_state.lines = pd.DataFrame(columns=['start_x', 'start_y', 'end_x', 'end_y', 'bearing', 'bearing_desc', 'distance', 'monument'])
            st.session_state.current_point = [0, 0]
            st.session_state.supplemental_info = None
            # Reset all line input fields
            for i in range(4):
                st.session_state[f"cardinal_ns_{i}"] = "North"
                st.session_state[f"degrees_{i}"] = 0
                st.session_state[f"minutes_{i}"] = 0
                st.session_state[f"seconds_{i}"] = 0
                st.session_state[f"cardinal_ew_{i}"] = "East"
                st.session_state[f"distance_{i}"] = 0.00
                st.session_state[f"monument_{i}"] = ""

    # Export DXF button
    with col4:
        if st.button("Export DXF", use_container_width=True):
            if not st.session_state.lines.empty:
                try:
                    dxf_data = create_dxf()
                    if dxf_data and len(dxf_data) > 0:
                        st.download_button(
                            label="Download DXF",
                            data=dxf_data,
                            file_name="line_drawing.dxf",
                            mime="application/octet-stream"
                        )
                    else:
                        st.error("Error: Generated DXF file is empty")
                except Exception as e:
                    st.error(f"Error creating DXF file: {str(e)}")
            else:
                st.warning("Add some lines before exporting")

    # New Export button
    with col5:
        if st.button("Export", use_container_width=True):
            if not st.session_state.lines.empty:
                try:
                    dxf_data = create_dxf()
                    if dxf_data and len(dxf_data) > 0:
                        st.download_button(
                            label="Download DXF",
                            data=dxf_data,
                            file_name="line_drawing.dxf",
                            mime="application/octet-stream"
                        )
                    else:
                        st.error("Error: Generated DXF file is empty")
                except Exception as e:
                    st.error(f"Error creating DXDXF file: {str(e)}")
            else:
                st.warning("Add some lines before exporting")

    # Display the plot
    fig = draw_lines()
    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)

    # Display supplemental information if available
    if st.session_state.supplemental_info:
        st.subheader("Property Information")
        if st.session_state.supplemental_info['land_lot']:
            st.write(f"Land Lot: {st.session_state.supplemental_info['land_lot']}")
        if st.session_state.supplemental_info['district']:
            st.write(f"District: {st.session_state.supplemental_info['district']}")
        if st.session_state.supplemental_info['county']:
            st.write(f"County: {st.session_state.supplemental_info['county']}")

    # Display PDF image if available
    if st.session_state.pdf_image:
        st.subheader("PDF Document")
        st.write("Please review your document shown below to verify the system correctly recognized the meets and bounds")
        st.image(st.session_state.pdf_image, caption="PDF First Page", use_container_width=True)

if __name__ == "__main__":
    main()