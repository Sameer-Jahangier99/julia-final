using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using Flux
using MLUtils
using Statistics
using Random
using JLD2
using CUDA
using ProgressMeter

# Set random seed for reproducibility
Random.seed!(42)

# Data loading and preprocessing functions
function load_anomaly_free_data(filepath)
    """Load normal data and add anomaly label"""
    df = CSV.read(filepath, DataFrame)
    
    # Add anomaly column (all zeros for normal data)
    df.anomaly = zeros(Float32, nrow(df))
    # Add changepoint column (all zeros for normal data)
    df.changepoint = zeros(Float32, nrow(df))
    
    return df
end

function load_anomaly_data(filepath)
    """Load anomaly data"""
    df = CSV.read(filepath, DataFrame)
    return df
end

function preprocess_data(df)
    """Extract features and labels, normalize features"""
    # Feature columns (excluding datetime, anomaly, changepoint)
    feature_cols = [:Accelerometer1RMS, :Accelerometer2RMS, :Current, :Pressure, 
                   :Temperature, :Thermocouple, :Voltage, :VolumeFlo]
    
    # Handle potential column name variations
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
    
    return Float32.(X), Float32.(y), actual_feature_cols[1:8]
end

function normalize_features(X_train, X_test=nothing)
    """Normalize features using training set statistics"""
    μ = mean(X_train, dims=1)
    σ = std(X_train, dims=1) .+ 1e-8  # Add small epsilon to avoid division by zero
    
    X_train_norm = (X_train .- μ) ./ σ
    
    if X_test !== nothing
        X_test_norm = (X_test .- μ) ./ σ
        return X_train_norm, X_test_norm, μ, σ
    else
        return X_train_norm, μ, σ
    end
end

function create_windowed_data(X, y, window_size)
    """Create sliding windows with statistical features"""
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
        
        # Calculate advanced statistical features for each sensor
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

function create_model(input_dim, max_params=1000)
    """Create a neural network model within parameter constraints"""
    
    println("Creating model for input_dim=$input_dim, max_params=$max_params")
    
    # With statistical features, input_dim should be around 40 (8 sensors × 4 stats + 8 current)
    # For 40 inputs and 1000 params, we can have much larger models
    
    # Calculate optimal hidden sizes for 2-layer network
    # Params = input_dim * h1 + h1 + h1 * 1 + 1 = h1 * (input_dim + 2) + 1
    max_h1 = div(max_params - 1, input_dim + 2)
    h1 = min(max_h1, 64)  # Cap at 64 for stability
    h1 = max(h1, 16)  # Minimum 16 neurons for capacity
    
    # Create 2-layer model
    model = Chain(
        Dense(input_dim, h1, relu),
        Dropout(0.3),
        Dense(h1, 1, sigmoid)
    )
    
    # Check parameter count - handle newer Flux API
    function count_model_params(model)
        total = 0
        for layer in model
            if isa(layer, Dense)
                total += length(layer.weight) + length(layer.bias)
            end
            # Skip Dropout layers as they have no parameters
        end
        return total
    end
    total_params = count_model_params(model)
    println("2-layer model: input->$h1->1, total_params=$total_params")
    
    # If we have room for a 3-layer network, try that
    if total_params < max_params * 0.7  # Use only 70% for 3-layer attempt
        h2 = div(h1, 2)
        h2 = max(h2, 8)  # Minimum 8 neurons
        
        # Try 3-layer model
        model_3layer = Chain(
            Dense(input_dim, h1, relu),
            Dropout(0.3),
            Dense(h1, h2, relu),
            Dropout(0.2),
            Dense(h2, 1, sigmoid)
        )
        
        total_params_3layer = count_model_params(model_3layer)
        println("3-layer model attempt: input->$h1->$h2->1, total_params=$total_params_3layer")
        
        if total_params_3layer <= max_params
            model = model_3layer
            total_params = total_params_3layer
            println("Using 3-layer model")
        else
            println("3-layer exceeds limit, using 2-layer model")
        end
    end
    
    println("Final model: $total_params parameters (limit: $max_params)")
    
    return model
end

function balanced_accuracy(y_true, y_pred)
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

function train_model(model, X_train, y_train, X_val, y_val; epochs=100, batch_size=128)
    """Train the model with specified hyperparameters"""
    
    # AdamW optimizer with specified parameters
    opt = AdamW(0.001, (0.9, 0.999), 0.001)  # lr=0.001, weight_decay=0.001
    opt_state = Flux.setup(opt, model)  # Setup optimizer state
    
    # Calculate class weights to handle imbalance
    n_normal = sum(y_train .== 0)
    n_anomaly = sum(y_train .== 1)
    total = n_normal + n_anomaly
    
    # Weight inversely proportional to class frequency
    weight_normal = total / (2 * n_normal)
    weight_anomaly = total / (2 * n_anomaly)
    
    println("Class weights: Normal=$(round(weight_normal, digits=3)), Anomaly=$(round(weight_anomaly, digits=3))")
    
    # Weighted binary cross-entropy loss
    function weighted_bce_loss(x, y)
        ŷ = vec(model(x))
        # Apply class weights
        weights = [yi == 0 ? weight_normal : weight_anomaly for yi in y]
        loss = -mean(weights .* (y .* log.(ŷ .+ 1f-8) .+ (1 .- y) .* log.(1 .- ŷ .+ 1f-8)))
        return loss
    end
    
    loss_fn = weighted_bce_loss
    
    # Training data loader
    train_data = DataLoader((X_train', y_train), batchsize=batch_size, shuffle=true)
    
    # Training history
    train_losses = Float32[]
    val_losses = Float32[]
    val_accuracies = Float32[]
    
    best_val_acc = 0.0
    best_model = deepcopy(model)
    
    @showprogress for epoch in 1:epochs
        # Training
        epoch_train_loss = 0.0
        num_batches = 0
        
        for (x_batch, y_batch) in train_data
            loss, grads = Flux.withgradient(model) do m
                loss_fn(x_batch, y_batch)
            end
            
            Flux.update!(opt_state, model, grads[1])
            
            epoch_train_loss += loss
            num_batches += 1
        end
        
        avg_train_loss = epoch_train_loss / num_batches
        push!(train_losses, avg_train_loss)
        
        # Validation
        val_pred = model(X_val')
        val_loss = Flux.Losses.binarycrossentropy(vec(val_pred), y_val)
        val_acc = balanced_accuracy(y_val, vec(val_pred))
        
        push!(val_losses, val_loss)
        push!(val_accuracies, val_acc)
        
        # Save best model
        if val_acc > best_val_acc
            best_val_acc = val_acc
            best_model = deepcopy(model)
        end
        
        # Print progress every 10 epochs
        if epoch % 10 == 0
            println("Epoch $epoch: Train Loss = $(round(avg_train_loss, digits=4)), Val Loss = $(round(val_loss, digits=4)), Val Acc = $(round(val_acc, digits=4))")
        end
    end
    
    return best_model, train_losses, val_losses, val_accuracies, best_val_acc
end

# Main execution
function main()
    println("Loading and preprocessing data...")
    
    # Load all data files
    println("Loading anomaly-free data...")
    normal_data = load_anomaly_free_data("data/anomaly-free/anomaly-free.csv")
    
    println("Loading anomaly data...")
    anomaly_files = [
        "data/valve1/0.csv", "data/valve1/1.csv", "data/valve1/2.csv",
        "data/valve2/0.csv",
        "data/other/1.csv", "data/other/2.csv", "data/other/3.csv"
    ]
    
    println("Total anomaly files to load: $(length(anomaly_files))")
    
    anomaly_data = []
    for file in anomaly_files
        if isfile(file)
            push!(anomaly_data, load_anomaly_data(file))
            println("Loaded $file")
        else
            println("Warning: File $file not found")
        end
    end
    
    # Combine all data
    all_data = vcat(normal_data, anomaly_data...)
    println("Combined dataset size: $(nrow(all_data)) samples")
    
    # Preprocess features
    X, y, feature_names = preprocess_data(all_data)
    println("Features: $feature_names")
    println("Feature matrix size: $(size(X))")
    println("Label distribution: Normal=$(sum(y .== 0)), Anomaly=$(sum(y .== 1))")
    
    # Better data split strategy for anomaly detection
    # Split normal and anomaly data separately to ensure representation
    normal_indices = findall(y .== 0)
    anomaly_indices = findall(y .== 1)
    
    println("Data distribution:")
    println("  Normal samples: $(length(normal_indices))")
    println("  Anomaly samples: $(length(anomaly_indices))")
    
    # Split normal data (80% train, 20% test)
    n_normal = length(normal_indices)
    normal_train_size = div(4 * n_normal, 5)
    normal_train_idx = normal_indices[1:normal_train_size]
    normal_test_idx = normal_indices[(normal_train_size+1):end]
    
    # Split anomaly data (80% train, 20% test)  
    n_anomaly = length(anomaly_indices)
    anomaly_train_size = div(4 * n_anomaly, 5)
    anomaly_train_idx = anomaly_indices[1:anomaly_train_size]
    anomaly_test_idx = anomaly_indices[(anomaly_train_size+1):end]
    
    # Combine training and test indices
    train_idx = vcat(normal_train_idx, anomaly_train_idx)
    test_idx = vcat(normal_test_idx, anomaly_test_idx)
    
    # Shuffle the training data
    train_idx = train_idx[randperm(length(train_idx))]
    
    X_train_raw, X_test_raw = X[train_idx, :], X[test_idx, :]
    y_train, y_test = y[train_idx], y[test_idx]
    
    println("After stratified split:")
    println("  Training: Normal=$(sum(y_train .== 0)), Anomaly=$(sum(y_train .== 1))")
    println("  Testing: Normal=$(sum(y_test .== 0)), Anomaly=$(sum(y_test .== 1))")
    
    # Normalize features
    X_train, X_test, μ, σ = normalize_features(X_train_raw, X_test_raw)
    
    println("Training set: $(size(X_train, 1)) samples")
    println("Test set: $(size(X_test, 1)) samples")
    
    # Window sizes to train
    window_sizes = [30, 90, 270]
    models = Dict()
    results = Dict()
    
    for window_size in window_sizes
        println("\n" * "="^50)
        println("Training model for window size: $window_size")
        println("="^50)
        
        # Create windowed data
        X_train_windowed, y_train_windowed = create_windowed_data(X_train, y_train, window_size)
        X_test_windowed, y_test_windowed = create_windowed_data(X_test, y_test, window_size)
        
        if X_train_windowed === nothing
            println("Skipping window size $window_size - not enough data")
            continue
        end
        
        println("Windowed training data: $(size(X_train_windowed))")
        println("Windowed test data: $(size(X_test_windowed))")
        
        # Train/validation split from training data
        n_train = size(X_train_windowed, 1)
        val_idx = 1:div(n_train, 5)  # 20% for validation
        train_idx = (div(n_train, 5)+1):n_train
        
        X_tr, X_val = X_train_windowed[train_idx, :], X_train_windowed[val_idx, :]
        y_tr, y_val = y_train_windowed[train_idx], y_train_windowed[val_idx]
        
        # Create model
        input_dim = size(X_tr, 2)
        model = create_model(input_dim)
        
        # Train model
        println("Training model...")
        trained_model, train_losses, val_losses, val_accs, best_val_acc = train_model(
            model, X_tr, y_tr, X_val, y_val, epochs=100, batch_size=128
        )
        
        # Test the model
        test_pred = trained_model(X_test_windowed')
        test_acc = balanced_accuracy(y_test_windowed, vec(test_pred))
        
        println("Best validation accuracy: $(round(best_val_acc, digits=4))")
        println("Test accuracy: $(round(test_acc, digits=4))")
        
        # Store results
        models[window_size] = trained_model
        results[window_size] = Dict(
            "train_losses" => train_losses,
            "val_losses" => val_losses,
            "val_accuracies" => val_accs,
            "best_val_acc" => best_val_acc,
            "test_acc" => test_acc,
            "μ" => μ,
            "σ" => σ,
            "feature_names" => feature_names
        )
        
        # Save model
        model_filename = "model_window_$(window_size).jld2"
        JLD2.@save model_filename trained_params=Flux.trainable(trained_model) trained_st=Flux.state(trained_model) μ=μ σ=σ feature_names=feature_names window_size=window_size
        println("Model saved as: $model_filename")
    end
    
    # Print final results
    println("\n" * "="^50)
    println("FINAL RESULTS")
    println("="^50)
    
    for window_size in window_sizes
        if haskey(results, window_size)
            result = results[window_size]
            println("Window size $window_size:")
            println("  Best validation accuracy: $(round(result["best_val_acc"], digits=4))")
            println("  Test accuracy: $(round(result["test_acc"], digits=4))")
        end
    end
    
    return models, results
end

# Run the main function
if abspath(PROGRAM_FILE) == @__FILE__
    models, results = main()
end