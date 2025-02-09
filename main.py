import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from math import radians
import ezdxf
from io import BytesIO, StringIO

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
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Add lines to the DXF
    for _, row in st.session_state.lines.iterrows():
        # Convert coordinates to float and create line
        start_point = (float(row['start_x']), float(row['start_y']), 0)
        end_point = (float(row['end_x']), float(row['end_y']), 0)
        msp.add_line(start_point, end_point)

    # Create string buffer first
    string_buffer = StringIO()
    doc.saveas(string_buffer)

    # Convert to bytes
    return string_buffer.getvalue().encode('utf-8')

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
            text=[f'Line {idx+1}: {row["bearing_desc"]}, {row["distance"]} units'],
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
    st.title("Line Drawing Application")
    initialize_session_state()

    # Input fields in a single row
    st.subheader("Add New Line")

    # Create columns for all inputs in a single row
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    with col1:
        cardinal_ns = st.selectbox("N/S", ["North", "South"], key="cardinal_ns")

    with col2:
        degrees = st.number_input("Deg", min_value=0, max_value=90, value=0, step=1, format="%d")

    with col3:
        minutes = st.number_input("Min", min_value=0, max_value=59, value=0, format="%d")

    with col4:
        seconds = st.number_input("Sec", min_value=0, max_value=59, value=0, format="%d")

    with col5:
        cardinal_ew = st.selectbox("E/W", ["East", "West"], key="cardinal_ew")

    with col6:
        distance = st.number_input("Distance", min_value=0.0, value=1.0, format="%.1f")

    with col7:
        if st.button("Add Line", use_container_width=True):
            if distance > 0:
                # Convert DMS to decimal degrees
                bearing = dms_to_decimal(degrees, minutes, seconds, cardinal_ns, cardinal_ew)

                # Calculate new endpoint
                end_point = calculate_endpoint(st.session_state.current_point, bearing, distance)

                # Create bearing description
                bearing_desc = f"{cardinal_ns} {degrees}° {minutes}' {seconds}\" {cardinal_ew}"

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
            else:
                st.error("Distance must be greater than 0")

    # Add a separator
    st.markdown("---")

    # Create a row for additional control buttons
    col1, col2, col3 = st.columns(3)

    # Add Rectangle button
    with col1:
        if st.button("Add Rectangle", use_container_width=True):
            if distance > 0:
                rectangle_lines = create_rectangle(st.session_state.current_point, distance)
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
                dxf_buffer = create_dxf()
                st.download_button(
                    label="Download DXF",
                    data=dxf_buffer,
                    file_name="line_drawing.dxf",
                    mime="application/octet-stream"
                )
            else:
                st.warning("Add some lines before exporting")

    # Display the plot
    fig = draw_lines()
    st.plotly_chart(fig, use_container_width=True)

    # Display line data
    if not st.session_state.lines.empty:
        st.subheader("Line Data")
        st.dataframe(st.session_state.lines[['bearing_desc', 'distance']])

if __name__ == "__main__":
    main()