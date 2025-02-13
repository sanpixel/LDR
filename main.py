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

def extract_bearings_from_text(text):
    """Extract bearings from legal description format."""
    try:
        # Find the legal description section
        start_marker = "beginning at a point"
        end_marker = "point of beginning"

        # Case insensitive search
        text_lower = text.lower()
        start_idx = text_lower.find(start_marker)
        end_idx = text_lower.find(end_marker)

        if start_idx == -1 or end_idx == -1:
            st.warning("Could not find complete legal description markers")
            st.info(f"Looking for text starting with '{start_marker}' and ending with '{end_marker}'")
            return []

        # Extract the relevant section
        legal_desc = text[start_idx:end_idx]

        # Split by 'thence' to get individual bearings
        segments = [s.strip() for s in legal_desc.split('thence')]

        # Display the segments for debugging
        st.subheader("Extracted Segments")
        for i, segment in enumerate(segments):
            st.text(f"Segment {i}:\n{segment}")

        bearings = []
        # Pattern matches formats like "North 11 degrees 22 minutes 33 seconds East"
        pattern = r'(North|South)\s+(\d+)\s*(?:degrees|°)?\s*(\d+)\s*(?:minutes|\'|′)?\s*(\d+)\s*(?:seconds|"|″)?\s+(East|West)'

        for segment in segments:
            # Look for bearing pattern in each segment
            match = re.search(pattern, segment, re.IGNORECASE)
            if match:
                cardinal_ns, deg, min, sec, cardinal_ew = match.groups()

                distance = 0.00 # Default distance
                distance_pattern = r'(\d+(?:\.\d+)?)\s*(?=feet)' # Retained original regex
                distance_match = re.search(distance_pattern, segment, re.IGNORECASE)
                if distance_match:
                    # Convert comma to period before converting to float
                    distance_str = distance_match.group(1).replace(',', '.')
                    distance = float(distance_str)


                bearings.append({
                    'cardinal_ns': 'North' if cardinal_ns.lower() == 'north' else 'South',
                    'degrees': int(deg),
                    'minutes': int(min),
                    'seconds': int(sec),
                    'cardinal_ew': 'East' if cardinal_ew.lower() == 'east' else 'West',
                    'distance': distance,
                    'original_text': segment.strip()
                })

        # Display what we found
        if bearings:
            st.subheader("Found Bearings and Distances")
            for i, bearing in enumerate(bearings):
                st.text(f"Bearing {i+1}:\n{bearing['original_text']}\nDistance: {bearing['distance']:.2f} feet")
        else:
            st.warning("No bearings found in the expected format")

        return bearings
    except Exception as e:
        st.error(f"Error parsing text: {str(e)}")
        return []

def process_pdf(uploaded_file):
    """Process uploaded PDF file and extract bearings."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        # Convert PDF to images
        images = convert_from_path(pdf_path)

        # Extract text from each page
        extracted_text = ""
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            extracted_text += f"\n--- Page {i+1} ---\n{text}\n"

        # Clean up temporary file
        os.unlink(pdf_path)

        # Display raw extracted text
        st.text_area("Raw Extracted Text", extracted_text, height=300)

        # Extract bearings from text
        bearings = extract_bearings_from_text(extracted_text)
        return bearings
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
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

        # Debug: Show how many lines we're processing
        st.write(f"Processing {len(st.session_state.lines)} lines for DXF export")

        # Add each line
        for idx, row in st.session_state.lines.iterrows():
            start = (float(row['start_x']), float(row['start_y']))
            end = (float(row['end_x']), float(row['end_y']))

            # Debug: Show coordinates being added
            st.write(f"Adding line {idx+1}: ({start[0]}, {start[1]}) to ({end[0]}, {end[1]})")

            try:
                msp.add_line((start[0], start[1]), (end[0], end[1]), dxfattribs={"layer": "Lines"})
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
        st.session_state.lines = pd.DataFrame(columns=['start_x', 'start_y', 'end_x', 'end_y', 'bearing', 'distance', 'bearing_desc'])
    if 'current_point' not in st.session_state:
        st.session_state.current_point = [0, 0]

def draw_lines():
    """Create a Plotly figure with all lines."""
    fig = go.Figure()

    # Draw all lines
    for idx, row in st.session_state.lines.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['start_x'], row['end_x']],
            y=[row['start_y'], row['end_y']],
            mode='lines+text',
            name=f'Line {idx+1}',
            text=[f'Line {idx+1}: {row["bearing_desc"]}, {row["distance"]:.2f} units'], #Updated to 2 decimals
            textposition='top center',
            line=dict(width=2)
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
        xaxis_title='X Coordinate',
        yaxis_title='Y Coordinate',
        xaxis=dict(
            zeroline=True,
            scaleanchor="y",
            scaleratio=1,
            constrain="domain"
        ),
        yaxis=dict(
            zeroline=True,
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
                        bearing = bearings[i]  # Get the correct bearing for this iteration
                        # Debug output for distance value
                        st.write(f"Debug: Bearing {i+1} distance value: {bearing['distance']}")

                        # Ensure all values are properly typed
                        st.session_state[f"cardinal_ns_{i}"] = bearing['cardinal_ns']
                        st.session_state[f"degrees_{i}"] = int(bearing['degrees'])
                        st.session_state[f"minutes_{i}"] = int(bearing['minutes'])
                        st.session_state[f"seconds_{i}"] = int(bearing['seconds'])
                        st.session_state[f"cardinal_ew_{i}"] = bearing['cardinal_ew']
                        st.session_state[f"distance_{i}"] = float(bearing['distance'])
                    else:
                        # Initialize remaining fields to defaults
                        st.session_state[f"cardinal_ns_{i}"] = "North"
                        st.session_state[f"degrees_{i}"] = 0
                        st.session_state[f"minutes_{i}"] = 0
                        st.session_state[f"seconds_{i}"] = 0
                        st.session_state[f"cardinal_ew_{i}"] = "East"
                        st.session_state[f"distance_{i}"] = 0.00

    # Line Drawing Section
    st.subheader("Draw Lines")

    # Create a container for all line inputs
    with st.container():
        for line_num in range(4):
            st.write(f"Line {line_num + 1}")
            col1, col2, col3, col4, col5, col6 = st.columns(6)

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

    # Control Buttons
    col1, col2, col3, col4 = st.columns(4)

    # Draw Lines button
    if st.button("Draw Lines", use_container_width=True):
        for line_num in range(4):
            # Only process lines with non-zero distance
            distance = st.session_state[f"distance_{line_num}"]
            if distance > 0:
                # Convert DMS to decimal degrees
                bearing = dms_to_decimal(
                    st.session_state[f"degrees_{line_num}"],
                    st.session_state[f"minutes_{line_num}"],
                    st.session_state[f"seconds_{line_num}"],
                    st.session_state[f"cardinal_ns_{line_num}"],
                    st.session_state[f"cardinal_ew_{line_num}"]
                )

                # Calculate new endpoint
                end_point = calculate_endpoint(st.session_state.current_point, bearing, distance)


                # Create bearing description
                bearing_desc = f"{st.session_state[f'cardinal_ns_{line_num}']} {st.session_state[f'degrees_{line_num}']}° {st.session_state[f'minutes_{line_num}']}' {st.session_state[f'seconds_{line_num}']}\" {st.session_state[f'cardinal_ew_{line_num}']}"

                # Add new line to DataFrame
                new_line = pd.DataFrame({
                    'start_x': [st.session_state.current_point[0]],
                    'start_y': [st.session_state.current_point[1]],
                    'end_x': [end_point[0]],
                    'end_y': [end_point[1]],
                    'bearing': [bearing],
                    'bearing_desc': [bearing_desc],
                    'distance': [distance]
                })
                st.session_state.lines = pd.concat([st.session_state.lines, new_line], ignore_index=True)

                # Update current point
                st.session_state.current_point = end_point

    # Add Rectangle button
    with col1:
        if st.button("Add Rectangle", use_container_width=True):
            if st.session_state.get(f"distance_0", 0) > 0:
                rectangle_lines = create_rectangle(st.session_state.current_point, st.session_state[f"distance_0"])
                st.session_state.lines = pd.concat([st.session_state.lines, rectangle_lines], ignore_index=True)
                st.session_state.current_point = [st.session_state.current_point[0], st.session_state.current_point[1]]
            else:
                st.error("Distance must be greater than 0")

    # Clear all button
    with col2:
        if st.button("Clear All", use_container_width=True):
            st.session_state.lines = pd.DataFrame(columns=['start_x', 'start_y', 'end_x', 'end_y', 'bearing', 'bearing_desc', 'distance'])
            st.session_state.current_point = [0, 0]

    # Export DXF button
    with col3:
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

    # Test DXF button
    with col4:
        if st.button("Test DXF", use_container_width=True):
            try:
                test_dxf_data = create_test_dxf()
                if test_dxf_data and len(test_dxf_data) > 0:
                    st.download_button(
                        label="Download Test DXF",
                        data=test_dxf_data,
                        file_name="test.dxf",
                        mime="application/octet-stream"
                    )
                else:
                    st.error("Error: Generated test DXF file is empty")
            except Exception as e:
                st.error(f"Error creating test DXF file: {str(e)}")

    # Display the plot
    fig = draw_lines()
    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)

    # Display line data
    if not st.session_state.lines.empty:
        st.subheader("Line Data")
        st.dataframe(st.session_state.lines[['bearing_desc', 'distance']])

if __name__ == "__main__":
    main()