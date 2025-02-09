import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from math import radians

def calculate_endpoint(start_point, bearing, distance):
    """Calculate endpoint coordinates given start point, bearing and distance."""
    bearing_rad = radians(bearing)
    dx = distance * np.sin(bearing_rad)
    dy = distance * np.cos(bearing_rad)
    end_x = start_point[0] + dx
    end_y = start_point[1] + dy
    return [end_x, end_y]

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'lines' not in st.session_state:
        st.session_state.lines = pd.DataFrame(columns=['start_x', 'start_y', 'end_x', 'end_y', 'bearing', 'distance'])
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
            text=[f'Line {idx+1}<br>Bearing: {row["bearing"]}°<br>Distance: {row["distance"]}'],
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

    # Update layout
    fig.update_layout(
        showlegend=True,
        title='Connected Lines Visualization',
        xaxis_title='X Coordinate',
        yaxis_title='Y Coordinate',
        xaxis=dict(zeroline=True),
        yaxis=dict(zeroline=True),
        aspectmode='equal'
    )

    return fig

def main():
    st.title("Line Drawing Application")
    initialize_session_state()

    # Create two columns for input and visualization
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input Parameters")
        
        # Input fields
        bearing = st.number_input("Bearing (degrees)", 
                                min_value=0.0, 
                                max_value=360.0, 
                                value=0.0,
                                step=1.0)
        
        distance = st.number_input("Distance", 
                                 min_value=0.0, 
                                 value=1.0,
                                 step=0.1)

        # Add line button
        if st.button("Add Line"):
            if distance > 0:
                # Calculate new endpoint
                end_point = calculate_endpoint(st.session_state.current_point, bearing, distance)
                
                # Add new line to DataFrame
                new_line = pd.DataFrame({
                    'start_x': [st.session_state.current_point[0]],
                    'start_y': [st.session_state.current_point[1]],
                    'end_x': [end_point[0]],
                    'end_y': [end_point[1]],
                    'bearing': [bearing],
                    'distance': [distance]
                })
                st.session_state.lines = pd.concat([st.session_state.lines, new_line], ignore_index=True)
                
                # Update current point
                st.session_state.current_point = end_point
            else:
                st.error("Distance must be greater than 0")

        # Clear all button
        if st.button("Clear All"):
            st.session_state.lines = pd.DataFrame(columns=['start_x', 'start_y', 'end_x', 'end_y', 'bearing', 'distance'])
            st.session_state.current_point = [0, 0]

        # Display line data
        st.subheader("Line Data")
        st.dataframe(st.session_state.lines[['bearing', 'distance']])

    with col2:
        # Display the plot
        fig = draw_lines()
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
