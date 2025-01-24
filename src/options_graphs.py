## import add_experiment_conditions from prepare graph data.
## looking at only non-reasoning models, let's create 3 visualisations to explore the options tasks:
# 
#  Here are a few complementary approaches:

# Per-Task Delta Analysis


# Calculate two deltas for each task:

# Δ1 = coordinate-no-reasoning - control
# Δ2 = coordinate-cot - coordinate-no-reasoning
# Use top_prop_all and top_prop_answered. Allow functions to take either of these as their metric of choice for the graph/s to produce

# This shows which tasks are most affected by each transition
# Chart 1: Visualize as a scatter plot with Δ1 vs Δ2


# Chart 2: Task Contribution Waterfall


# Sort tasks by their contribution to overall performance changes
# Show cumulative impact of each task
# Helps identify which tasks are driving the major differences


# Chart 3: Condition-Task Interaction Plot


# Create a plot with:

# X-axis: Tasks (ordered by some metric)
# Y-axis: Performance
# Three lines for the conditions
# This shows where conditions diverge most