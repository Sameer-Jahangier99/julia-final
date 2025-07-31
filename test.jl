using CSV
using DataFrames
using Flux
using Statistics
using JLD2

function load_test_data()
    """Load and preprocess test data - all anomaly files"""
    
    # Load all anomaly files for testing
    anomaly_files = [
        "data/valve1/0.csv", "data/valve1/1.csv", "data/valve1/2.csv",
        "data/valve2/0.csv",
        "data/other/1.csv", "data/other/2.csv", "data/other/3.csv"
    ]
    
    anomaly_data = []
    for file in anomaly_files
        if isfile(file)
            df = CSV.read(file, DataFrame)
            push!(anomaly_data, df)
            println("Loaded test data: $file")
        else
            println("Warning: Test file $file not found")
        end
    end
    
    # Combine all anomaly data
    all_test_data = vcat(anomaly_data...)
    
    return all_test_data
end

function preprocess_test_data(df, feature_names)
    """Extract and preprocess features from test data"""
    
    # Handle potential column name variations - same as training
    actual_feature_cols = []
    for col in names(df)
        if occursin("Accelerometer1", col) push!(actual_feature_cols, col) end
        if occursin("Accelerometer2", col) push!(actual_feature_cols, col) end
        if occursin("Current", col) push!(actual_feature_cols, col) end
        if occursin("Pressure", col) push!(actual_feature_cols, col) end
        if occursin("Temperature", col) && !occursin("Thermocouple", col) push!(actual_feature_cols, col) end
        if occursin("Thermocouple", col) push!(actual_feature_cols, col) end
        if occursin("Voltage", col) push!(actual_feature_cols, col) end
        if occursin("Volume", col) || occursin("Flow", col) push!(actual_feature_cols, col) end
    end
    
    # Extract features and labels
    X = Matrix(df[:, actual_feature_cols[1:8]])  # Get first 8 feature columns
    y = Vector(df.anomaly)
    
    return Float32.(X), Float32.(y)
end

function normalize_test_features(X_test, μ, σ)
    """Normalize test features using training set statistics"""
    X_test_norm = (X_test .- μ) ./ σ
    return X_test_norm
end

function create_windowed_test_data(X, y, window_size)
    """Create sliding windows with statistical features for test data"""
    n_samples, n_features = size(X)
    
    if n_samples <= window_size
        @warn "Not enough samples for window size $window_size"
        return nothing, nothing
    end
    
    # Create windowed features using statistics instead of raw values
    X_windowed = []
    y_windowed = []
    
    for i in (window_size + 1):n_samples
        # Extract window: past window_size + current timestep
        window_data = X[(i-window_size):i, :]  # window_size + 1 timesteps
        
        # Calculate advanced statistical features for each sensor (same as training)
        window_features = Float32[]
        
        for j in 1:n_features
            sensor_data = window_data[:, j]
            
            # Basic statistics
            push!(window_features, mean(sensor_data))
            push!(window_features, std(sensor_data))
            
            # Range and spread indicators (good for detecting anomalies)
            push!(window_features, maximum(sensor_data) - minimum(sensor_data))  # Range
            
            # Trend indicators (detect gradual changes)
            if length(sensor_data) > 1
                # Linear trend (slope of best fit line)
                x_vals = 1:length(sensor_data)
                trend = (length(sensor_data) * sum(x_vals .* sensor_data) - sum(x_vals) * sum(sensor_data)) / 
                       (length(sensor_data) * sum(x_vals.^2) - sum(x_vals)^2)
                push!(window_features, trend)
            else
                push!(window_features, 0.0f0)
            end
            
            # Deviation from start of window (detect sudden changes)
            push!(window_features, sensor_data[end] - sensor_data[1])
        end
        
        # Add current timestep raw values for immediate context
        append!(window_features, X[i, :])
        
        push!(X_windowed, window_features)
        push!(y_windowed, y[i])  # Label for current timestep
    end
    
    X_out = hcat(X_windowed...)'  # Convert to matrix
    y_out = Vector(y_windowed)
    
    return Float32.(X_out), Float32.(y_out)
end

function balanced_accuracy_metric(y_true, y_pred)
    """Calculate balanced accuracy"""
    y_pred_binary = y_pred .>= 0.5
    
    # True positives, false positives, true negatives, false negatives
    tp = sum((y_true .== 1) .& (y_pred_binary .== 1))
    fp = sum((y_true .== 0) .& (y_pred_binary .== 1))
    tn = sum((y_true .== 0) .& (y_pred_binary .== 0))
    fn = sum((y_true .== 1) .& (y_pred_binary .== 0))
    
    # Sensitivity (True Positive Rate)
    sensitivity = tp / (tp + fn + 1e-8)
    
    # Specificity (True Negative Rate)  
    specificity = tn / (tn + fp + 1e-8)
    
    # Balanced accuracy
    bal_acc = (sensitivity + specificity) / 2
    
    return bal_acc
end

function bal_acc(model_path, test_data=nothing)
    """
    Main function to calculate balanced accuracy as required by the task.
    
    This function:
    1. Loads the saved model from JLD2 file
    2. Loads and preprocesses test data
    3. Applies the same transformations used during training
    4. Makes predictions and calculates balanced accuracy
    
    Args:
        model_path: Path to the saved JLD2 model file
        test_data: Optional test data (if not provided, loads from data files)
    
    Returns:
        Balanced accuracy score
    """
    
    println("Loading model from: $model_path")
    
    # Load the saved model and preprocessing parameters
    JLD2.@load model_path trained_params trained_st μ σ feature_names window_size
    
    println("Model window size: $window_size")
    println("Feature names: $feature_names")
    
    # Reconstruct the model architecture based on saved parameters
    println("Saved parameters structure: ", keys(trained_params))
    
    # For our models, we know the structure: input->hidden->1 with dropout
    # Get dimensions from first layer
    first_layer = trained_params.layers[1]  # First Dense layer
    
    input_dim = size(first_layer.weight, 2)
    hidden_size = size(first_layer.weight, 1)
    
    println("Reconstructed model: input_dim=$input_dim, hidden_size=$hidden_size")
    
    # Create model with same architecture as training
    model = Chain(
        Dense(input_dim, hidden_size, relu),
        Dropout(0.3),
        Dense(hidden_size, 1, sigmoid)
    )
    
    # Load the trained parameters directly
    model = Flux.loadmodel!(model, trained_params)
    
    # Load test data if not provided
    if test_data === nothing
        println("Loading test data...")
        test_df = load_test_data()
    else
        test_df = test_data
    end
    
    # Preprocess test data
    println("Preprocessing test data...")
    X_test_raw, y_test = preprocess_test_data(test_df, feature_names)
    
    # Apply normalization using training set statistics
    X_test_normalized = normalize_test_features(X_test_raw, μ, σ)
    
    # Create windowed data
    println("Creating windowed test data (window size: $window_size)...")
    X_test_windowed, y_test_windowed = create_windowed_test_data(X_test_normalized, y_test, window_size)
    
    if X_test_windowed === nothing
        error("Not enough test data for window size $window_size")
    end
    
    println("Test data shape: $(size(X_test_windowed))")
    println("Test labels shape: $(size(y_test_windowed))")
    println("Test label distribution: Normal=$(sum(y_test_windowed .== 0)), Anomaly=$(sum(y_test_windowed .== 1))")
    
    # Make predictions
    println("Making predictions...")
    y_pred = model(X_test_windowed')
    y_pred_vec = vec(y_pred)
    
    # Calculate balanced accuracy
    bal_acc_score = balanced_accuracy_metric(y_test_windowed, y_pred_vec)
    
    println("Balanced Accuracy: $(round(bal_acc_score, digits=4))")
    
    # Additional metrics for debugging
    y_pred_binary = y_pred_vec .>= 0.5
    accuracy = sum(y_test_windowed .== y_pred_binary) / length(y_test_windowed)
    println("Regular Accuracy: $(round(accuracy, digits=4))")
    
    return bal_acc_score
end

function test_all_models()
    """Test all saved models and compare their performance"""
    
    window_sizes = [30, 90, 270]
    results = Dict()
    
    println("="^60)
    println("TESTING ALL MODELS")
    println("="^60)
    
    for window_size in window_sizes
        model_filename = "model_window_$(window_size).jld2"
        
        if isfile(model_filename)
            println("\n" * "-"^40)
            println("Testing model for window size: $window_size")
            println("-"^40)
            
            try
                bal_acc_score = bal_acc(model_filename)
                results[window_size] = bal_acc_score
                
                println("✓ Model window_$window_size: Balanced Accuracy = $(round(bal_acc_score, digits=4))")
                
            catch e
                println("✗ Error testing model window_$window_size: $e")
                results[window_size] = nothing
            end
        else
            println("✗ Model file not found: $model_filename")
            results[window_size] = nothing
        end
    end
    
    # Summary
    println("\n" * "="^60)
    println("FINAL TEST RESULTS SUMMARY")
    println("="^60)
    
    best_window = nothing
    best_score = 0.0
    
    for window_size in window_sizes
        if haskey(results, window_size) && results[window_size] !== nothing
            score = results[window_size]
            println("Window size $window_size: Balanced Accuracy = $(round(score, digits=4))")
            
            if score > best_score
                best_score = score
                best_window = window_size
            end
        else
            println("Window size $window_size: Failed")
        end
    end
    
    if best_window !== nothing
        println("\n🏆 Best performing model: Window size $best_window with Balanced Accuracy = $(round(best_score, digits=4))")
    end
    
    return results
end

# Example usage and testing
if abspath(PROGRAM_FILE) == @__FILE__
    # Test all models
    results = test_all_models()
    
    # Example of testing a specific model (as required by task)
    println("\n" * "="^40)
    println("EXAMPLE: Testing specific model")
    println("="^40)
    
    # Test the first available model
    window_sizes = [30, 90, 270]
    for window_size in window_sizes
        model_path = "model_window_$(window_size).jld2"
        if isfile(model_path)
            println("Testing model: $model_path")
            
            # This is the format required by the task
            # JLD2.@load joinpath("$(path_to_saved_model).jld2") trained_params trained_st
            # bal_acc = bal_acc(**args, trained_params, trained_st, test_x, test_y)
            
            bal_acc_score = bal_acc(model_path)
            println("Final balanced accuracy: $(round(bal_acc_score, digits=4))")
            break
        end
    end
end
