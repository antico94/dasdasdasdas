"""
HTML Report Generator for Gold Trading Model Analysis.
This module generates a comprehensive HTML report from the analysis results.
"""

import os
import json
from datetime import datetime
import base64
from pathlib import Path


def get_badge_class(value, thresholds):
    """Return the appropriate badge class based on value and thresholds."""
    if value >= thresholds['good']:
        return "badge-success"
    elif value >= thresholds['average']:
        return "badge-warning"
    else:
        return "badge-danger"


def get_text_class(value, thresholds):
    """Return the appropriate text class based on value and thresholds."""
    if value >= thresholds['good']:
        return "metric-good"
    elif value >= thresholds['average']:
        return "metric-average"
    else:
        return "metric-poor"


def embed_image(image_path):
    """Embed image as base64 in HTML."""
    if not os.path.exists(image_path):
        return ""

    # Read image file
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Get file extension
    file_ext = os.path.splitext(image_path)[1].lower()[1:]  # Remove the dot
    if file_ext == 'jpg':
        file_ext = 'jpeg'

    return f"data:image/{file_ext};base64,{encoded_image}"


def model_info_section(model_info):
    """Generate the model information section."""
    # Extract metadata for easy access
    metadata = model_info.get('metadata', {})
    timeframe = model_info.get('timeframe', metadata.get('timeframe', 'Unknown'))
    prediction_horizon = model_info.get('prediction_horizon', metadata.get('prediction_horizon', 1))
    prediction_target = metadata.get('prediction_target', 'direction')

    # Format timeframe for display
    timeframe_display = timeframe
    if timeframe == 'H1':
        timeframe_display = 'H1 (1 Hour)'
    elif timeframe == 'D1':
        timeframe_display = 'D1 (Daily)'
    elif timeframe == 'M5':
        timeframe_display = 'M5 (5 Minutes)'

    # Convert prediction target for display
    target_display = "Price Direction (Up/Down)"
    if prediction_target == 'return':
        target_display = "Price Return (Percentage)"

    # Format data period
    data_period = model_info.get('data_period', {})
    start_date = data_period.get('start', 'Unknown')
    end_date = data_period.get('end', 'Unknown')

    # Create the metrics
    metrics_html = f"""
    <div class="metric">
        <div class="metric-label">Model Name:</div>
        <div class="metric-value">{model_info.get('model_name', 'Unknown')}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Created On:</div>
        <div class="metric-value">{model_info.get('creation_date', 'Unknown')}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Timeframe:</div>
        <div class="metric-value">{timeframe_display}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Prediction Target:</div>
        <div class="metric-value">{target_display}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Prediction Horizon:</div>
        <div class="metric-value">{prediction_horizon} {timeframe} period(s) ahead</div>
    </div>
    <div class="metric">
        <div class="metric-label">Testing Period:</div>
        <div class="metric-value">{start_date} - {end_date}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Number of Predictions:</div>
        <div class="metric-value">{model_info.get('data_rows', 'Unknown')}</div>
    </div>
    """

    # Add feature information if available
    if 'features' in metadata:
        features_str = ", ".join(metadata['features'])
        metrics_html += f"""
        <div class="metric">
            <div class="metric-label">Features Used:</div>
            <div class="metric-value">{len(metadata['features'])} features</div>
        </div>
        <div class="card-explanation">
            <p><strong>Features:</strong> {features_str}</p>
        </div>
        """

    # Build the complete section
    html = f"""
    <div class="card">
        <div class="card-header">Model Information</div>
        <div class="card-body">
            {metrics_html}

            <div class="card-explanation">
                <p><strong>What is this model?</strong> This model predicts whether gold prices will move up or down in the next {prediction_horizon} {timeframe} period(s). It was trained on historical gold price data with various technical indicators.</p>
            </div>
        </div>
    </div>
    """

    return html


def accuracy_section(accuracy_results, confusion_matrix_path):
    """Generate the accuracy analysis section."""
    if not accuracy_results:
        return ""

    # Extract data
    overall_accuracy = accuracy_results.get('overall_accuracy', 0)
    class_accuracy = accuracy_results.get('class_accuracy', {})

    # Set class labels
    class_labels = {0: "DOWN", 1: "UP"}

    # Define accuracy thresholds for badges
    accuracy_thresholds = {'good': 0.60, 'average': 0.53}

    # Determine badge for overall accuracy
    overall_class = get_text_class(overall_accuracy, accuracy_thresholds)

    # Create metrics HTML
    metrics_html = f"""
    <div class="metric">
        <div class="metric-label">Overall Accuracy:</div>
        <div class="metric-value {overall_class}">{overall_accuracy:.2%} 
            <span class="tooltip">ⓘ<span class="tooltiptext">The percentage of all predictions that correctly matched the actual price movement. Random guessing would be 50%.</span></span>
        </div>
    </div>
    """

    # Add class-specific metrics
    for class_id, label in class_labels.items():
        if str(class_id) in class_accuracy or class_id in class_accuracy:
            # Handle both string and int keys
            class_data = class_accuracy.get(str(class_id), class_accuracy.get(class_id, {}))
            accuracy = class_data.get('accuracy', 0)
            samples = class_data.get('samples', 0)
            class_text_class = get_text_class(accuracy, accuracy_thresholds)

            metrics_html += f"""
            <div class="metric">
                <div class="metric-label">Accuracy for {label}:</div>
                <div class="metric-value {class_text_class}">{accuracy:.2%} ({samples:,} samples)
                    <span class="tooltip">ⓘ<span class="tooltiptext">When the price actually went {label.lower()}, the model correctly predicted it this percentage of the time.</span></span>
                </div>
            </div>
            """

    # Add recommendations based on class accuracy differences
    recommendations = []
    up_accuracy = class_accuracy.get('1', class_accuracy.get(1, {})).get('accuracy', 0)
    down_accuracy = class_accuracy.get('0', class_accuracy.get(0, {})).get('accuracy', 0)

    # Analyze the bias
    bias_explanation = ""
    if up_accuracy > down_accuracy + 0.15:
        bias_explanation = "The model has a strong bias toward predicting upward movements (bull bias). It's much better at catching upward moves than downward ones."
        recommendations.append(
            "Consider using this model primarily for long (buy) signals due to its strength in predicting upward movements.")
        recommendations.append("For downward movements, you might need additional confirmation from other indicators.")
    elif down_accuracy > up_accuracy + 0.15:
        bias_explanation = "The model has a strong bias toward predicting downward movements (bear bias). It's much better at catching downward moves than upward ones."
        recommendations.append(
            "Consider using this model primarily for short (sell) signals due to its strength in predicting downward movements.")
        recommendations.append("For upward movements, you might need additional confirmation from other indicators.")
    else:
        bias_explanation = "The model shows balanced prediction accuracy between upward and downward movements."
        recommendations.append(
            "This model performs similarly for both up and down predictions, making it suitable for two-way trading.")

    # Add overall accuracy recommendations
    if overall_accuracy < 0.53:
        recommendations.append(
            "The overall accuracy is relatively low. Consider retraining the model or using it in combination with other signals.")
    elif overall_accuracy > 0.60:
        recommendations.append(
            "The model shows good overall accuracy. It can be used as a primary signal generator for trading decisions.")

    # Create recommendations HTML
    recommendations_html = ""
    if recommendations:
        recommendations_html = """
        <div class="recommendations">
            <h3>Recommendations</h3>
            <ul>
        """
        for rec in recommendations:
            recommendations_html += f"<li>{rec}</li>"
        recommendations_html += """
            </ul>
        </div>
        """

    # Embed confusion matrix image
    confusion_matrix_img = ""
    if confusion_matrix_path and os.path.exists(confusion_matrix_path):
        embedded_img = embed_image(confusion_matrix_path)
        confusion_matrix_img = f"""
        <div class="chart-container">
            <h3>Confusion Matrix</h3>
            <img src="{embedded_img}" alt="Confusion Matrix" style="max-width: 100%;">
            <div class="card-explanation">
                <p><strong>How to interpret:</strong> The confusion matrix shows the counts of correct and incorrect predictions. The vertical axis shows the actual values, while the horizontal axis shows the predicted values.</p>
                <ul>
                    <li>Top-left: True Negatives (correctly predicted DOWN)</li>
                    <li>Top-right: False Positives (incorrectly predicted UP)</li>
                    <li>Bottom-left: False Negatives (incorrectly predicted DOWN)</li>
                    <li>Bottom-right: True Positives (correctly predicted UP)</li>
                </ul>
            </div>
        </div>
        """

    # Build the complete section
    html = f"""
    <div class="card">
        <div class="card-header">Overall Prediction Accuracy</div>
        <div class="card-body">
            {metrics_html}

            <div class="card-explanation">
                <p><strong>What does this mean?</strong> {bias_explanation}</p>
            </div>

            {confusion_matrix_img}

            {recommendations_html}
        </div>
    </div>
    """

    return html


def confidence_section(accuracy_results, confidence_plot_path):
    """Generate the confidence analysis section."""
    if not accuracy_results or 'confidence' not in accuracy_results:
        return ""

    confidence_data = accuracy_results.get('confidence', {})
    confidence_stats = confidence_data.get('stats', {})
    confidence_by_level = confidence_data.get('by_level', {})

    # Create confidence stats metrics
    stats_html = f"""
    <div class="metric">
        <div class="metric-label">Minimum Confidence:</div>
        <div class="metric-value">{confidence_stats.get('min', 0):.4f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Maximum Confidence:</div>
        <div class="metric-value">{confidence_stats.get('max', 0):.4f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Average Confidence:</div>
        <div class="metric-value">{confidence_stats.get('mean', 0):.4f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Standard Deviation:</div>
        <div class="metric-value">{confidence_stats.get('std', 0):.4f}</div>
    </div>
    """

    # Create confidence by level table
    confidence_table_rows = ""
    confidence_increasing = True
    last_accuracy = 0

    # Sort confidence levels
    sorted_levels = sorted(confidence_by_level.keys(), key=lambda x: float(x.split(',')[0].strip('()')))

    for level in sorted_levels:
        level_data = confidence_by_level[level]
        accuracy = level_data.get('accuracy', 0)
        samples = level_data.get('samples', 0)
        percentage = level_data.get('percentage', 0)

        # Check if accuracy increases with confidence
        if accuracy < last_accuracy and last_accuracy > 0:
            confidence_increasing = False
        last_accuracy = accuracy

        # Get appropriate color class based on accuracy
        accuracy_class = get_text_class(accuracy, {'good': 0.60, 'average': 0.53})

        confidence_table_rows += f"""
        <tr>
            <td>{level}</td>
            <td class="{accuracy_class}">{accuracy:.2%}</td>
            <td>{samples:,}</td>
            <td>{percentage:.1f}%</td>
        </tr>
        """

    confidence_table = f"""
    <table class="table">
        <thead>
            <tr>
                <th>Confidence Range</th>
                <th>Accuracy</th>
                <th>Samples</th>
                <th>% of Total</th>
            </tr>
        </thead>
        <tbody>
            {confidence_table_rows}
        </tbody>
    </table>
    """

    # Add recommendations based on confidence analysis
    confidence_recommendations = []

    # Check if higher confidence correlates with higher accuracy
    if confidence_increasing:
        confidence_recommendations.append(
            "Higher confidence predictions are more accurate. Consider filtering signals to only use those with higher confidence.")
    else:
        confidence_recommendations.append(
            "Higher confidence doesn't always mean higher accuracy. The model may be overconfident in some cases.")

    # Check confidence distribution
    max_conf = confidence_stats.get('max', 0)
    if max_conf < 0.7:
        confidence_recommendations.append(
            "The model's confidence is generally low, suggesting it's cautious in its predictions.")
    elif max_conf > 0.8:
        confidence_recommendations.append(
            "The model shows high confidence in some predictions, which can be used to identify strong trading signals.")

    # Create recommendations HTML
    confidence_recs_html = ""
    if confidence_recommendations:
        confidence_recs_html = """
        <div class="recommendations">
            <h3>Confidence Analysis Recommendations</h3>
            <ul>
        """
        for rec in confidence_recommendations:
            confidence_recs_html += f"<li>{rec}</li>"
        confidence_recs_html += """
            </ul>
        </div>
        """

    # Embed confidence plot
    confidence_plot_html = ""
    if confidence_plot_path and os.path.exists(confidence_plot_path):
        embedded_img = embed_image(confidence_plot_path)
        confidence_plot_html = f"""
        <div class="chart-container">
            <h3>Accuracy by Confidence Level</h3>
            <img src="{embedded_img}" alt="Accuracy by Confidence Level" style="max-width: 100%;">
        </div>
        """

    # Explanation of confidence
    confidence_explanation = """
    <div class="card-explanation">
        <p><strong>What is Prediction Confidence?</strong> Confidence represents how certain the model is about each prediction. A higher confidence value (closer to 1.0) means the model is more certain about its prediction.</p>
        <p><strong>How to interpret this data:</strong></p>
        <ul>
            <li>Ideally, higher confidence predictions should have higher accuracy.</li>
            <li>If high-confidence predictions aren't more accurate, the model may be overconfident.</li>
            <li>The distribution of confidence values shows if your model is generally cautious or bold in its predictions.</li>
        </ul>
        <p><strong>Trading implications:</strong> Consider setting a minimum confidence threshold (e.g., 0.65) for actual trading signals to filter out low-confidence predictions.</p>
    </div>
    """

    # Build the complete section
    html = f"""
    <div class="card">
        <div class="card-header">Confidence Analysis</div>
        <div class="card-body">
            {stats_html}

            {confidence_explanation}

            <h3>Accuracy by Confidence Level</h3>
            {confidence_table}

            {confidence_plot_html}

            {confidence_recs_html}
        </div>
    </div>
    """

    return html


def consecutive_runs_section(consecutive_results, runs_plot_path):
    """Generate the consecutive runs analysis section."""
    if not consecutive_results:
        return ""

    run_stats = consecutive_results.get('run_stats', {})

    # Add explanation
    runs_explanation = """
    <div class="card-explanation">
        <p><strong>What are Consecutive Prediction Runs?</strong> A "run" is a sequence of consecutive predictions of the same type (UP or DOWN) made by the model. For example, if the model predicts UP 5 times in a row before switching to DOWN, that's a run length of 5.</p>
        <p><strong>Why is this important?</strong> Run length distribution helps you understand the model's behavior over time:</p>
        <ul>
            <li><strong>Short runs</strong> (1-3 periods): The model frequently changes its mind, which could lead to excessive trading.</li>
            <li><strong>Medium runs</strong> (4-10 periods): More stable predictions that might align with short-term market trends.</li>
            <li><strong>Long runs</strong> (10+ periods): The model is detecting persistent market conditions or has a strong bias in one direction.</li>
        </ul>
        <p><strong>Trading implications:</strong> You might want to use longer runs as stronger signals and ignore very short runs to reduce false signals.</p>
    </div>
    """

    # Create summary table
    summary_table = """
    <table class="table">
        <thead>
            <tr>
                <th>Prediction Type</th>
                <th>Average Run Length</th>
                <th>Maximum Run Length</th>
                <th>Number of Runs</th>
                <th>Most Common Run Length</th>
            </tr>
        </thead>
        <tbody>
    """

    for pred_type, stats in run_stats.items():
        # Find most common run length
        distribution = stats.get('distribution', {})
        most_common_length = 0
        most_common_count = 0

        for length, count in distribution.items():
            if count > most_common_count:
                most_common_length = length
                most_common_count = count

        pred_label = "UP" if pred_type == "1" or pred_type == 1 else "DOWN"

        summary_table += f"""
        <tr>
            <td>{pred_label}</td>
            <td>{stats.get('avg_length', 0):.2f}</td>
            <td>{stats.get('max_length', 0)}</td>
            <td>{stats.get('num_runs', 0)}</td>
            <td>{most_common_length} ({most_common_count} occurrences)</td>
        </tr>
        """

    summary_table += """
        </tbody>
    </table>
    """

    # Create run distribution visualization
    distribution_viz = """
    <div class="run-distribution">
        <h3>Run Length Distribution</h3>
        <div class="row">
    """

    # Create simplified distribution for common run lengths
    for pred_type, stats in run_stats.items():
        distribution = stats.get('distribution', {})
        pred_label = "UP" if pred_type == "1" or pred_type == 1 else "DOWN"

        # Group distribution into meaningful bins
        bins = {
            "1-3 (Very Short)": 0,
            "4-10 (Short)": 0,
            "11-20 (Medium)": 0,
            "21-50 (Long)": 0,
            "51+ (Very Long)": 0
        }

        for length, count in distribution.items():
            length = int(length)
            if length <= 3:
                bins["1-3 (Very Short)"] += count
            elif length <= 10:
                bins["4-10 (Short)"] += count
            elif length <= 20:
                bins["11-20 (Medium)"] += count
            elif length <= 50:
                bins["21-50 (Long)"] += count
            else:
                bins["51+ (Very Long)"] += count

        # Create progress bars
        total_runs = stats.get('num_runs', 0)
        progress_bars = ""

        for bin_name, count in bins.items():
            if total_runs > 0:
                percentage = count / total_runs * 100

                # Determine color based on bin
                color = "#3498db"  # Default blue
                if "Very Short" in bin_name:
                    color = "#e74c3c"  # Red
                elif "Short" in bin_name:
                    color = "#f39c12"  # Orange
                elif "Medium" in bin_name:
                    color = "#2ecc71"  # Green
                elif "Long" in bin_name:
                    color = "#27ae60"  # Darker green
                elif "Very Long" in bin_name:
                    color = "#16a085"  # Teal

                progress_bars += f"""
                <div class="metric">
                    <div class="metric-label">{bin_name}:</div>
                    <div class="metric-value">{count} runs ({percentage:.1f}%)</div>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-bar" style="width: {percentage}%; background-color: {color};">{percentage:.1f}%</div>
                </div>
                """

        distribution_viz += f"""
        <div class="col-md-6">
            <h4>{pred_label} Predictions</h4>
            {progress_bars}
        </div>
        """

    distribution_viz += """
        </div>
    </div>
    """

    # Embed runs plot
    runs_plot_html = ""
    if runs_plot_path and os.path.exists(runs_plot_path):
        embedded_img = embed_image(runs_plot_path)
        runs_plot_html = f"""
        <div class="chart-container">
            <h3>Detailed Run Length Distribution</h3>
            <img src="{embedded_img}" alt="Run Length Distribution" style="max-width: 100%;">
            <p><em>This plot shows the detailed distribution of run lengths for both UP and DOWN predictions.</em></p>
        </div>
        """

    # Add recommendations based on run analysis
    run_recommendations = []

    # Analyze UP runs
    up_stats = run_stats.get("1", run_stats.get(1, {}))
    up_avg_length = up_stats.get('avg_length', 0)

    # Analyze DOWN runs
    down_stats = run_stats.get("0", run_stats.get(0, {}))
    down_avg_length = down_stats.get('avg_length', 0)

    # Compare UP and DOWN runs
    if up_avg_length > down_avg_length * 2:
        run_recommendations.append(
            "The model has much longer UP prediction runs than DOWN runs, suggesting a strong bullish bias.")
    elif down_avg_length > up_avg_length * 2:
        run_recommendations.append(
            "The model has much longer DOWN prediction runs than UP runs, suggesting a strong bearish bias.")

    # Check for very short runs
    up_distribution = up_stats.get('distribution', {})
    down_distribution = down_stats.get('distribution', {})

    short_up_runs = sum(up_distribution.get(str(i), 0) for i in range(1, 4))
    short_down_runs = sum(down_distribution.get(str(i), 0) for i in range(1, 4))

    if short_up_runs / up_stats.get('num_runs', 1) > 0.5 or short_down_runs / down_stats.get('num_runs', 1) > 0.5:
        run_recommendations.append(
            "The model frequently produces very short prediction runs, which could lead to excessive trading. Consider using a filter to require multiple consecutive signals in the same direction.")

    # Check for very long runs
    long_up_runs = sum(up_distribution.get(str(i), 0) for i in range(21, 1000))
    long_down_runs = sum(down_distribution.get(str(i), 0) for i in range(21, 1000))

    if long_up_runs > 0 or long_down_runs > 0:
        run_recommendations.append(
            "The model sometimes produces very long prediction runs. These might represent strong trends and could be valuable trading opportunities.")

    # Create recommendations HTML
    run_recs_html = ""
    if run_recommendations:
        run_recs_html = """
        <div class="recommendations">
            <h3>Run Analysis Recommendations</h3>
            <ul>
        """
        for rec in run_recommendations:
            run_recs_html += f"<li>{rec}</li>"
        run_recs_html += """
            </ul>
        </div>
        """

    # Build the complete section
    html = f"""
    <div class="card">
        <div class="card-header">Consecutive Predictions Analysis</div>
        <div class="card-body">
            {runs_explanation}

            <h3>Run Length Summary</h3>
            {summary_table}

            {distribution_viz}

            {runs_plot_html}

            {run_recs_html}
        </div>
    </div>
    """

    return html


def price_movement_section(price_analysis, price_plot_path):
    """Generate the price movement analysis section."""
    if not price_analysis:
        return ""

    price_changes = price_analysis.get('price_changes', {})
    volatility = price_analysis.get('volatility', {})

    # Create price changes table
    price_table = """
    <table class="table">
        <thead>
            <tr>
                <th>Prediction Type</th>
                <th>Mean Price Change</th>
                <th>Standard Deviation</th>
                <th>Sample Count</th>
            </tr>
        </thead>
        <tbody>
    """

    for pred_type, stats in price_changes.items():
        pred_label = "UP" if pred_type == "1" or pred_type == 1 else "DOWN"
        mean_change = stats.get('mean', 0)
        std_dev = stats.get('std', 0)
        count = stats.get('count', 0)

        # Determine class for mean change
        mean_class = ""
        if (pred_label == "UP" and mean_change > 0) or (pred_label == "DOWN" and mean_change < 0):
            mean_class = "metric-good"
        elif abs(mean_change) < 0.1:
            mean_class = "metric-average"
        else:
            mean_class = "metric-poor"

        price_table += f"""
        <tr>
            <td>{pred_label}</td>
            <td class="{mean_class}">{mean_change:.4f}%</td>
            <td>{std_dev:.4f}%</td>
            <td>{count:,}</td>
        </tr>
        """

    price_table += """
        </tbody>
    </table>
    """

    # Explanation of price changes
    price_explanation = """
    <div class="card-explanation">
        <p><strong>What does this mean?</strong> This analysis shows how actual price movements correspond to the model's predictions. Ideally:</p>
        <ul>
            <li>When the model predicts UP, the average price change should be positive</li>
            <li>When the model predicts DOWN, the average price change should be negative</li>
        </ul>
        <p>The standard deviation indicates the variability in price changes. A high standard deviation means the actual price changes can vary significantly even when the model makes the same prediction.</p>
    </div>
    """

    # Embed price plot
    price_plot_html = ""
    if price_plot_path and os.path.exists(price_plot_path):
        embedded_img = embed_image(price_plot_path)
        price_plot_html = f"""
        <div class="chart-container">
            <h3>Distribution of Price Changes by Prediction Type</h3>
            <img src="{embedded_img}" alt="Price Changes by Prediction Type" style="max-width: 100%;">
        </div>
        """

    # Create volatility table
    volatility_table = """
    <table class="table">
        <thead>
            <tr>
                <th>Volatility Range</th>
                <th>Accuracy</th>
                <th>Sample Count</th>
            </tr>
        </thead>
        <tbody>
    """

    # Sort volatility ranges
    sorted_volatility = sorted(volatility.keys(), key=lambda x: float(x.split(',')[0].strip('()')))

    for vol_range in sorted_volatility:
        vol_data = volatility[vol_range]
        accuracy = vol_data.get('accuracy', 0)
        samples = vol_data.get('samples', 0)

        # Get appropriate color class based on accuracy
        accuracy_class = get_text_class(accuracy, {'good': 0.60, 'average': 0.53})

        volatility_table += f"""
        <tr>
            <td>{vol_range}</td>
            <td class="{accuracy_class}">{accuracy:.2%}</td>
            <td>{samples:,}</td>
        </tr>
        """

    volatility_table += """
        </tbody>
    </table>
    """

    # Add volatility explanation
    volatility_explanation = """
    <div class="card-explanation">
        <p><strong>What is the Volatility Analysis?</strong> This analysis shows how well the model performs under different market volatility conditions. Volatility is measured as the range between high and low prices relative to the low price.</p>
        <p><strong>How to interpret this data:</strong></p>
        <ul>
            <li><strong>Low volatility</strong>: Markets are calm with smaller price movements.</li>
            <li><strong>High volatility</strong>: Markets are experiencing large price swings, possibly due to news or uncertainty.</li>
        </ul>
        <p><strong>Trading implications:</strong> If the model performs significantly better in certain volatility conditions, you might want to only trade those conditions or adjust position sizing accordingly.</p>
    </div>
    """

    # Add recommendations based on price analysis
    price_recommendations = []

    # Check if predictions align with actual price movements
    up_stats = price_changes.get("1", price_changes.get(1, {}))
    down_stats = price_changes.get("0", price_changes.get(0, {}))

    up_mean = up_stats.get('mean', 0)
    down_mean = down_stats.get('mean', 0)

    if up_mean <= 0:
        price_recommendations.append(
            "When the model predicts UP, prices actually tend to move DOWN on average. This suggests the model's UP predictions might be more valuable as contrarian indicators.")

    if down_mean >= 0:
        price_recommendations.append(
            "When the model predicts DOWN, prices actually tend to move UP on average. This suggests the model's DOWN predictions might be more valuable as contrarian indicators.")

    # Analyze volatility performance
    best_volatility_range = None
    best_accuracy = 0

    for vol_range, vol_data in volatility.items():
        acc = vol_data.get('accuracy', 0)
        if acc > best_accuracy:
            best_accuracy = acc
            best_volatility_range = vol_range

    if best_volatility_range and best_accuracy > 0.55:
        price_recommendations.append(
            f"The model performs best in volatility range {best_volatility_range} with {best_accuracy:.2%} accuracy. Consider focusing on trades during these volatility conditions.")

    # Create recommendations HTML
    price_recs_html = ""
    if price_recommendations:
        price_recs_html = """
        <div class="recommendations">
            <h3>Price Movement Analysis Recommendations</h3>
            <ul>
        """
        for rec in price_recommendations:
            price_recs_html += f"<li>{rec}</li>"
        price_recs_html += """
            </ul>
        </div>
        """

    # Build the complete section
    html = f"""
    <div class="card">
        <div class="card-header">Price Movement Analysis</div>
        <div class="card-body">
            <h3>Price Changes by Prediction Type</h3>
            {price_table}

            {price_explanation}

            {price_plot_html}

            <h3>Accuracy by Volatility Level</h3>
            {volatility_table}

            {volatility_explanation}

            {price_recs_html}
        </div>
    </div>
    """

    return html


def trading_opportunities_section(trading_results):
    """Generate the trading opportunities section."""
    if not trading_results:
        return ""

    # Extract data
    confidence_threshold = trading_results.get('confidence_threshold', 0.65)
    signals_summary = trading_results.get('signals_summary', {})
    recent_signals = trading_results.get('recent_signals', [])
    latest_signal = trading_results.get('latest_signal', {})

    # Create signals summary
    up_signals = signals_summary.get('up', {})
    down_signals = signals_summary.get('down', {})

    up_count = up_signals.get('count', 0)
    up_percentage = up_signals.get('percentage', 0)
    down_count = down_signals.get('count', 0)
    down_percentage = down_signals.get('percentage', 0)

    signals_summary_html = f"""
    <div class="metric">
        <div class="metric-label">Confidence Threshold:</div>
        <div class="metric-value">{confidence_threshold:.2f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">High-confidence UP signals (last 50 periods):</div>
        <div class="metric-value">{up_count} ({up_percentage:.1f}%)</div>
    </div>
    <div class="metric">
        <div class="metric-label">High-confidence DOWN signals (last 50 periods):</div>
        <div class="metric-value">{down_count} ({down_percentage:.1f}%)</div>
    </div>
    """

    # Create recent signals table
    recent_signals_table = """
    <table class="table">
        <thead>
            <tr>
                <th>Timestamp</th>
                <th>Direction</th>
                <th>Confidence</th>
                <th>Outcome</th>
            </tr>
        </thead>
        <tbody>
    """

    for signal in recent_signals:
        timestamp = signal.get('timestamp', '')
        direction = signal.get('direction', '')
        confidence = signal.get('confidence', 0)
        correct = signal.get('correct')

        # Determine badge class based on direction
        direction_class = "badge-success" if direction == "UP" else "badge-danger"

        # Determine outcome text and class
        outcome_text = "Unknown"
        outcome_class = ""

        if correct is not None:
            if correct:
                outcome_text = "Correct ✓"
                outcome_class = "metric-good"
            else:
                outcome_text = "Incorrect ✗"
                outcome_class = "metric-poor"

        recent_signals_table += f"""
        <tr>
            <td>{timestamp}</td>
            <td><span class="badge {direction_class}">{direction}</span></td>
            <td>{confidence:.4f}</td>
            <td class="{outcome_class}">{outcome_text}</td>
        </tr>
        """

    recent_signals_table += """
        </tbody>
    </table>
    """

    # Latest signal box
    latest_signal_html = ""
    if latest_signal:
        timestamp = latest_signal.get('timestamp', '')
        direction = latest_signal.get('direction', '')
        confidence = latest_signal.get('confidence', 0)
        is_opportunity = latest_signal.get('is_trading_opportunity', False)

        # Determine direction class
        direction_class = "success" if direction == "UP" else "danger"

        # Create HTML based on whether it's a trading opportunity
        if is_opportunity:
            latest_signal_html = f"""
            <div class="trading-opportunity">
                <h3>Trading Opportunity Detected</h3>
                <div class="metric">
                    <div class="metric-label">Signal:</div>
                    <div class="metric-value"><span class="badge badge-{direction_class}">{direction}</span></div>
                </div>
                <div class="metric">
                    <div class="metric-label">Timestamp:</div>
                    <div class="metric-value">{timestamp}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Confidence:</div>
                    <div class="metric-value">{confidence:.4f}</div>
                </div>
                <p>This signal meets your confidence threshold of {confidence_threshold} and may represent a trading opportunity.</p>
            </div>
            """
        else:
            latest_signal_html = f"""
            <div class="card-explanation">
                <h3>Latest Signal - Not a Trading Opportunity</h3>
                <div class="metric">
                    <div class="metric-label">Signal:</div>
                    <div class="metric-value"><span class="badge badge-{direction_class}">{direction}</span></div>
                </div>
                <div class="metric">
                    <div class="metric-label">Timestamp:</div>
                    <div class="metric-value">{timestamp}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Confidence:</div>
                    <div class="metric-value">{confidence:.4f}</div>
                </div>
                <p>This signal does not meet your confidence threshold of {confidence_threshold} and is not considered a high-confidence trading opportunity.</p>
            </div>
            """

    # Add explanation
    trading_explanation = """
    <div class="card-explanation">
        <p><strong>What are Trading Opportunities?</strong> Trading opportunities are identified based on high-confidence predictions from the model. By filtering for higher confidence levels, you can focus on signals that the model is more certain about.</p>
        <p><strong>How to use this information:</strong></p>
        <ul>
            <li>High-confidence signals are those that meet or exceed your confidence threshold.</li>
            <li>Recent signals give you an idea of the model's recent performance and current trend prediction.</li>
            <li>The latest signal shows the most recent prediction and whether it qualifies as a trading opportunity.</li>
        </ul>
    </div>
    """

    # Build the complete section
    html = f"""
    <div class="card">
        <div class="card-header">Potential Trading Opportunities</div>
        <div class="card-body">
            {signals_summary_html}

            {trading_explanation}

            <h3>Recent Signals (Last 5 Periods)</h3>
            {recent_signals_table}

            {latest_signal_html}
        </div>
    </div>
    """

    return html


def summary_section(results):
    """Generate overall summary and recommendations section."""
    # Extract overall metrics
    model_info = results.get('model_info', {})
    accuracy_analysis = results.get('accuracy_analysis', {})
    time_analysis = results.get('time_analysis', {})
    consecutive_analysis = results.get('consecutive_analysis', {})
    price_analysis = results.get('price_analysis', {})
    trading_opportunities = results.get('trading_opportunities', {})

    overall_accuracy = accuracy_analysis.get('overall_accuracy', 0)

    # Create summary metrics
    summary_metrics = f"""
    <div class="metric">
        <div class="metric-label">Model:</div>
        <div class="metric-value">{model_info.get('model_name', 'Unknown')}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Overall Accuracy:</div>
        <div class="metric-value {get_text_class(overall_accuracy, {'good': 0.60, 'average': 0.53})}">{overall_accuracy:.2%}</div>
    </div>
    """

    # Add metrics for recent period if available
    daily_accuracy = time_analysis.get('daily_accuracy', {})
    if daily_accuracy:
        # Get last 10 days average
        recent_days = sorted(daily_accuracy.keys())[-10:]
        recent_accuracy = sum(daily_accuracy[day] for day in recent_days) / len(recent_days) if recent_days else 0

        summary_metrics += f"""
        <div class="metric">
            <div class="metric-label">Recent 10 Days Average Accuracy:</div>
            <div class="metric-value {get_text_class(recent_accuracy, {'good': 0.60, 'average': 0.53})}">{recent_accuracy:.2%}</div>
        </div>
        """

    # Add latest signal info if available
    latest_signal = trading_opportunities.get('latest_signal', {})
    if latest_signal:
        direction = latest_signal.get('direction', '')
        confidence = latest_signal.get('confidence', 0)
        direction_class = "badge-success" if direction == "UP" else "badge-danger"

        summary_metrics += f"""
        <div class="metric">
            <div class="metric-label">Latest Signal:</div>
            <div class="metric-value"><span class="badge {direction_class}">{direction}</span> (Confidence: {confidence:.4f})</div>
        </div>
        """

    # Collect all recommendations
    all_recommendations = []

    # Add key insights based on analysis
    insights = []

    # Accuracy insights
    class_accuracy = accuracy_analysis.get('class_accuracy', {})
    up_accuracy = class_accuracy.get('1', class_accuracy.get(1, {})).get('accuracy', 0)
    down_accuracy = class_accuracy.get('0', class_accuracy.get(0, {})).get('accuracy', 0)

    if abs(up_accuracy - down_accuracy) > 0.2:
        if up_accuracy > down_accuracy:
            insights.append(
                f"Strong bull bias: {up_accuracy:.2%} accuracy for UP predictions vs {down_accuracy:.2%} for DOWN")
            all_recommendations.append("Use this model primarily for identifying buying opportunities")
        else:
            insights.append(
                f"Strong bear bias: {down_accuracy:.2%} accuracy for DOWN predictions vs {up_accuracy:.2%} for UP")
            all_recommendations.append("Use this model primarily for identifying selling opportunities")

    # Run length insights
    run_stats = consecutive_analysis.get('run_stats', {})
    if run_stats:
        up_stats = run_stats.get("1", run_stats.get(1, {}))
        down_stats = run_stats.get("0", run_stats.get(0, {}))

        up_avg_length = up_stats.get('avg_length', 0)
        down_avg_length = down_stats.get('avg_length', 0)

        if max(up_avg_length, down_avg_length) < 3:
            insights.append(
                f"Short prediction runs (UP: {up_avg_length:.1f}, DOWN: {down_avg_length:.1f}) indicate frequent signal changes")
            all_recommendations.append(
                "Consider using a filter to require multiple consecutive signals to reduce noise")
        elif max(up_avg_length, down_avg_length) > 8:
            insights.append(
                f"Long prediction runs (UP: {up_avg_length:.1f}, DOWN: {down_avg_length:.1f}) indicate stable signals")
            all_recommendations.append("The model may be effective at detecting sustained trends")

    # Price movement insights
    price_changes = price_analysis.get('price_changes', {})
    if price_changes:
        up_stats = price_changes.get("1", price_changes.get(1, {}))
        down_stats = price_changes.get("0", price_changes.get(0, {}))

        up_mean = up_stats.get('mean', 0)
        down_mean = down_stats.get('mean', 0)

        if (up_mean > 0 and down_mean < 0):
            insights.append(f"Price movements align with predictions (UP: {up_mean:.2f}%, DOWN: {down_mean:.2f}%)")
            all_recommendations.append("The model's directional predictions have economic value")
        elif (up_mean <= 0 or down_mean >= 0):
            insights.append(f"Price movements contradict predictions (UP: {up_mean:.2f}%, DOWN: {down_mean:.2f}%)")
            all_recommendations.append("Consider using the model as a contrarian indicator")

    # Create insights HTML
    insights_html = "<ul>"
    for insight in insights:
        insights_html += f"<li>{insight}</li>"
    insights_html += "</ul>"

    # Create recommendations HTML
    recommendations_html = "<ul>"
    for rec in all_recommendations:
        recommendations_html += f"<li>{rec}</li>"
    recommendations_html += "</ul>"

    # Build the complete section
    html = f"""
    <div class="card">
        <div class="card-header">Executive Summary</div>
        <div class="card-body">
            {summary_metrics}

            <h3>Key Insights</h3>
            {insights_html}

            <h3>Overall Recommendations</h3>
            {recommendations_html}
        </div>
    </div>
    """

    return html


def generate_html_report(results, output_path):
    """Generate a comprehensive HTML report from analysis results."""
    # Get paths to plots
    time_analysis = results.get('time_analysis', {})
    accuracy_analysis = results.get('accuracy_analysis', {})
    consecutive_analysis = results.get('consecutive_analysis', {})
    price_analysis = results.get('price_analysis', {})

    daily_accuracy_plot = time_analysis.get('daily_accuracy_plot', '')
    probability_plot = time_analysis.get('probability_plot', '')
    confusion_matrix_plot = accuracy_analysis.get('confusion_matrix_plot', '')
    confidence_plot = accuracy_analysis.get('confidence_plot', '')
    runs_plot = consecutive_analysis.get('runs_plot', '')
    price_change_plot = price_analysis.get('price_change_plot', '')

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build HTML report with all sections
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Gold Trading Model Performance Report</title>
        <style>
            :root {{
                --primary-color: #1a5276;
                --secondary-color: #f39c12;
                --background-color: #f8f9fa;
                --card-bg: white;
                --text-color: #333;
                --border-color: #ddd;
                --success-color: #27ae60;
                --warning-color: #e67e22;
                --danger-color: #c0392b;
                --info-color: #3498db;
            }}

            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: var(--text-color);
                background-color: var(--background-color);
                margin: 0;
                padding: 0;
            }}

            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}

            .header {{
                background-color: var(--primary-color);
                color: white;
                padding: 20px;
                text-align: center;
                border-radius: 8px 8px 0 0;
                margin-bottom: 30px;
            }}

            .header h1 {{
                margin: 0;
                font-size: 28px;
            }}

            .header p {{
                margin: 10px 0 0;
                opacity: 0.8;
            }}

            .card {{
                background-color: var(--card-bg);
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                margin-bottom: 30px;
                overflow: hidden;
            }}

            .card-header {{
                background-color: var(--primary-color);
                color: white;
                padding: 15px 20px;
                font-weight: bold;
                font-size: 18px;
                border-bottom: 1px solid var(--border-color);
            }}

            .card-body {{
                padding: 20px;
            }}

            .card-explanation {{
                background-color: rgba(52, 152, 219, 0.1);
                border-left: 4px solid var(--info-color);
                padding: 15px;
                margin: 15px 0;
                border-radius: 0 8px 8px 0;
            }}

            .metric {{
                display: flex;
                align-items: center;
                margin-bottom: 15px;
            }}

            .metric-label {{
                width: 250px;
                font-weight: bold;
            }}

            .metric-value {{
                flex-grow: 1;
            }}

            .metric-good {{
                color: var(--success-color);
                font-weight: bold;
            }}

            .metric-average {{
                color: var(--warning-color);
                font-weight: bold;
            }}

            .metric-poor {{
                color: var(--danger-color);
                font-weight: bold;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}

            table, th, td {{
                border: 1px solid var(--border-color);
            }}

            th {{
                background-color: #f2f2f2;
                padding: 12px;
                text-align: left;
            }}

            td {{
                padding: 10px;
            }}

            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}

            .chart-container {{
                margin: 20px 0;
                max-width: 100%;
            }}

            .summary-box {{
                background-color: #f8f9fa;
                border: 1px solid var(--border-color);
                border-radius: 8px;
                padding: 20px;
                margin-top: 20px;
            }}

            .summary-title {{
                font-weight: bold;
                margin-bottom: 10px;
                color: var(--primary-color);
            }}

            .tooltip {{
                position: relative;
                display: inline-block;
                cursor: help;
            }}

            .tooltip .tooltiptext {{
                visibility: hidden;
                width: 300px;
                background-color: #555;
                color: #fff;
                text-align: left;
                border-radius: 6px;
                padding: 10px;
                position: absolute;
                z-index: 1;
                bottom: 125%;
                left: 50%;
                margin-left: -150px;
                opacity: 0;
                transition: opacity 0.3s;
                font-weight: normal;
                font-size: 14px;
                line-height: 1.4;
            }}

            .tooltip:hover .tooltiptext {{
                visibility: visible;
                opacity: 1;
            }}

            .badge {{
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
                margin-left: 8px;
            }}

            .badge-success {{
                background-color: var(--success-color);
                color: white;
            }}

            .badge-warning {{
                background-color: var(--warning-color);
                color: white;
            }}

            .badge-danger {{
                background-color: var(--danger-color);
                color: white;
            }}

            .recommendations {{
                background-color: #ebf5fb;
                border-left: 4px solid var(--primary-color);
                padding: 15px;
                margin: 20px 0;
                border-radius: 0 8px 8px 0;
            }}

            .recommendations h3 {{
                margin-top: 0;
                color: var(--primary-color);
            }}

            .recommendations ul {{
                margin-bottom: 0;
            }}

            .tab {{
                overflow: hidden;
                border: 1px solid #ccc;
                background-color: #f1f1f1;
                border-radius: 8px 8px 0 0;
            }}

            .tab button {{
                background-color: inherit;
                float: left;
                border: none;
                outline: none;
                cursor: pointer;
                padding: 14px 16px;
                transition: 0.3s;
                font-size: 17px;
            }}

            .tab button:hover {{
                background-color: #ddd;
            }}

            .tab button.active {{
                background-color: var(--primary-color);
                color: white;
            }}

            .tabcontent {{
                display: none;
                padding: 20px;
                border: 1px solid #ccc;
                border-top: none;
                border-radius: 0 0 8px 8px;
                animation: fadeEffect 1s;
            }}

            @keyframes fadeEffect {{
                from {{opacity: 0;}}
                to {{opacity: 1;}}
            }}

            .progress-bar-container {{
                width: 100%;
                background-color: #e0e0e0;
                border-radius: 4px;
                margin: 8px 0;
            }}

            .progress-bar {{
                height: 20px;
                border-radius: 4px;
                text-align: center;
                line-height: 20px;
                color: white;
                font-weight: bold;
            }}

            .trading-opportunity {{
                background-color: rgba(39, 174, 96, 0.1);
                border: 1px solid var(--success-color);
                border-radius: 8px;
                padding: 15px;
                margin: 20px 0;
            }}

            .trading-opportunity h3 {{
                color: var(--success-color);
                margin-top: 0;
            }}

            footer {{
                text-align: center;
                margin-top: 30px;
                padding: 20px;
                color: #777;
                font-size: 14px;
                border-top: 1px solid var(--border-color);
            }}

            .col-md-6 {{
                width: 48%;
                display: inline-block;
                vertical-align: top;
                margin-right: 2%;
            }}

            .row {{
                display: flex;
                flex-wrap: wrap;
            }}

            /* Make responsive */
            @media (max-width: 768px) {{
                .col-md-6 {{
                    width: 100%;
                    margin-right: 0;
                }}

                .metric {{
                    flex-direction: column;
                    align-items: flex-start;
                }}

                .metric-label {{
                    width: 100%;
                    margin-bottom: 5px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Gold Trading Model Performance Report</h1>
                <p>Comprehensive analysis of model predictions and trading opportunities</p>
                <p><small>Generated on: {timestamp}</small></p>
            </div>

            {summary_section(results)}

            {model_info_section(results.get('model_info', {}))}

            {accuracy_section(results.get('accuracy_analysis', {}), confusion_matrix_plot)}

            {confidence_section(results.get('accuracy_analysis', {}), confidence_plot)}

            {consecutive_runs_section(results.get('consecutive_analysis', {}), runs_plot)}

            {price_movement_section(results.get('price_analysis', {}), price_change_plot)}

            {trading_opportunities_section(results.get('trading_opportunities', {}))}

            <footer>
                <p>Gold Trading Model Analysis Report | Generated by GoldML Analysis Tool</p>
            </footer>
        </div>

        <script>
            // Add interactive features
            document.addEventListener('DOMContentLoaded', function() {{
                // Make tooltips work
                const tooltips = document.querySelectorAll('.tooltip');
                tooltips.forEach(tooltip => {{
                    tooltip.addEventListener('mouseenter', function() {{
                        const tooltiptext = this.querySelector('.tooltiptext');
                        tooltiptext.style.visibility = 'visible';
                        tooltiptext.style.opacity = '1';
                    }});
                    tooltip.addEventListener('mouseleave', function() {{
                        const tooltiptext = this.querySelector('.tooltiptext');
                        tooltiptext.style.visibility = 'hidden';
                        tooltiptext.style.opacity = '0';
                    }});
                }});
            }});
        </script>
    </body>
    </html>
    """

    # Ensure directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write the report to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)

    return output_path