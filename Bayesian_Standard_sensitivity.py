#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adjusted code to run both the standard and Bayesian models with varying initial sensitivities,
run simulations on the same instances, and compare results using box plots.

Generates separate sets of box plots for each model and sensitivity value.
@author ozgunbaris
"""

import random
import math
import numpy as np
import pandas as pd
from scipy.special import lambertw  # Import Lambert W function
from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt
import seaborn as sns

prof_driver_cost = 5  # Reference cost (used for initial c_i calculation)
min_compensation = 2.5  # Define the minimum compensation

# Define the Task class
class Task:
    def __init__(self, task_id, x, y, start_time, end_time, delivery_x, delivery_y):
        self.task_id = task_id
        self.x = x  # x-coordinate of pickup location
        self.y = y  # y-coordinate of pickup location
        self.start_time = start_time  # Time when the task becomes available
        self.end_time = end_time  # Time when the task expires
        self.delivery_x = delivery_x  # x-coordinate of delivery location
        self.delivery_y = delivery_y  # y-coordinate of delivery location
        self.delivery_distance = self.compute_delivery_distance()
        self.available = True  # Task availability status

    def compute_delivery_distance(self):
        # Euclidean distance between pickup and delivery locations
        return math.hypot(self.delivery_x - self.x, self.delivery_y - self.y)

    def get_current_c(self, current_period):
        periods_remaining = self.end_time - current_period
        reduction = min(0.1 * periods_remaining * prof_driver_cost, 0.5 * prof_driver_cost)
        c = max(min_compensation, prof_driver_cost - reduction)
        return c

    def to_dict(self):
        return {
            'task_id': self.task_id,
            'x': self.x,
            'y': self.y,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'delivery_x': self.delivery_x,
            'delivery_y': self.delivery_y
        }

    @staticmethod
    def from_dict(data):
        return Task(
            data['task_id'],
            data['x'],
            data['y'],
            data['start_time'],
            data['end_time'],
            data['delivery_x'],
            data['delivery_y']
        )

# Define the Bayesian Driver class
class BayesianDriver:
    def __init__(self, driver_id, x, y, end_time, initial_sensitivity):
        self.driver_id = driver_id
        self.x = x  # x-coordinate
        self.y = y  # y-coordinate
        self.end_time = end_time  # Time when the driver becomes unavailable
        self.available = True  # Driver availability status

        # Bayesian updating parameters
        self.alpha = 1  # Prior success (acceptance)
        self.beta = 1   # Prior failure (rejection)
        self.initial_sensitivity = initial_sensitivity  # Initial sensitivity value
        self.sensitivity = self.calculate_sensitivity()

    def calculate_sensitivity(self):
        # Calculate expected acceptance probability
        expected_acceptance = self.alpha / (self.alpha + self.beta)
        # Map expected acceptance to sensitivity parameter
        # Sensitivity affects acceptance probability inversely
        # Adjust scaling as needed
        sensitivity = max(0.5, 2 * (1 - expected_acceptance))
        return sensitivity

    def update_parameters_bayesian(self, accepted):
        if accepted:
            self.alpha += 1
        else:
            self.beta += 1
        # Update sensitivity based on new alpha and beta
        self.sensitivity = self.calculate_sensitivity()

    def update_parameters_standard(self):
        # In the standard model, sensitivity does not update at all.
        # We keep sensitivity constant = initial_sensitivity
        self.sensitivity = self.initial_sensitivity if self.initial_sensitivity is not None else 1.0

    def to_dict(self):
        return {
            'driver_id': self.driver_id,
            'x': self.x,
            'y': self.y,
            'end_time': self.end_time,
            'initial_sensitivity': self.initial_sensitivity,
            'alpha': self.alpha,
            'beta': self.beta
        }

    @staticmethod
    def from_dict(data):
        driver = BayesianDriver(
            data['driver_id'],
            data['x'],
            data['y'],
            data['end_time'],
            data['initial_sensitivity']
        )
        driver.alpha = data.get('alpha', 1)
        driver.beta = data.get('beta', 1)
        driver.sensitivity = driver.calculate_sensitivity()
        return driver

# Function to generate instances
def generate_instances(n_periods, t_arrival_rate, d_arrival_rate, t_avg_time, d_avg_time, seed):
    random.seed(seed)
    np.random.seed(seed)
    total_task_no = 0
    total_driver_no = 0
    tasks = []
    drivers = []

    for p in range(1, n_periods + 1):
        # Generate number of tasks and drivers using Poisson distribution
        n_tasks = np.random.poisson(t_arrival_rate)
        n_drivers = np.random.poisson(d_arrival_rate)

        # Generate tasks
        for _ in range(n_tasks):
            total_task_no += 1
            task_id = total_task_no
            x = round(random.uniform(0, 200), 2)
            y = round(random.uniform(0, 200), 2)
            # Generate delivery coordinates
            delivery_x = round(random.uniform(0, 200), 2)
            delivery_y = round(random.uniform(0, 200), 2)
            # Generate duration from Poisson distribution with mean 3
            duration = np.random.poisson(3) + 1  # Ensure duration is at least 1
            start_time = p
            end_time = p + duration - 1
            task = Task(task_id, x, y, start_time, end_time, delivery_x, delivery_y)
            tasks.append(task)

        # Generate drivers
        for _ in range(n_drivers):
            total_driver_no += 1
            driver_id = total_driver_no
            x = round(random.uniform(0, 200), 2)
            y = round(random.uniform(0, 200), 2)
            time = np.random.poisson(d_avg_time)
            end_time = time + p
            # Initial sensitivity will be set during simulation
            initial_sensitivity = None
            driver = BayesianDriver(driver_id, x, y, end_time, initial_sensitivity)
            drivers.append(driver)

    return tasks, drivers

# Function to save instances to CSV
def save_instances_to_csv(tasks, drivers, instance_number):
    tasks_df = pd.DataFrame([task.to_dict() for task in tasks])
    drivers_df = pd.DataFrame([driver.to_dict() for driver in drivers])

    tasks_df.to_csv(f'tasks_instance_{instance_number}.csv', index=False)
    drivers_df.to_csv(f'drivers_instance_{instance_number}.csv', index=False)

# Function to read instances from CSV and set initial sensitivity
def read_instances_from_csv(instance_number, initial_sensitivity):
    tasks_df = pd.read_csv(f'tasks_instance_{instance_number}.csv')
    drivers_df = pd.read_csv(f'drivers_instance_{instance_number}.csv')

    tasks = [Task.from_dict(row) for _, row in tasks_df.iterrows()]

    # Set the initial sensitivity value for all drivers
    drivers = []
    for _, row in drivers_df.iterrows():
        driver_data = row.to_dict()
        driver_data['initial_sensitivity'] = initial_sensitivity
        driver = BayesianDriver.from_dict(driver_data)
        drivers.append(driver)

    return tasks, drivers

# Probability function
def probability_function(C_ij, sensitivity_j, delivery_distance_i, task_distance_ij):
    if C_ij <= 0:
        return 0.0  # Zero probability when compensation is zero or negative

    gamma = -1.451977
    delta = 0.04330204
    beta1 = 1.247782    # Coefficient for sensitivity
    beta2 = -0.01852485 # Coefficient for delivery distance
    beta3 = -3.893085   # Coefficient for task distance

    # Scale distances
    max_distance = math.hypot(200, 200)  # Approximately 283
    delivery_distance_scaled = delivery_distance_i / max_distance
    task_distance_scaled = task_distance_ij / max_distance

    exponent = -(gamma + delta * C_ij + beta1 * sensitivity_j + beta2 * delivery_distance_scaled + beta3 * task_distance_scaled)
    # Handle potential overflow in exponentials
    if exponent > 700:
        return 1.0
    elif exponent < -700:
        return 0.0
    else:
        return 1 / (1 + math.exp(exponent))

# Function to compute C^*_{ij} using the Lambert W function
def compute_cij_star(tasks, drivers, current_period):
    c_star = np.zeros((len(tasks), len(drivers)))
    P_star = np.zeros((len(tasks), len(drivers)))
    C_star_matrix = np.zeros((len(tasks), len(drivers)))
    for i, task in enumerate(tasks):
        delivery_distance_i = task.delivery_distance
        c_i = task.get_current_c(current_period)
        for j, driver in enumerate(drivers):
            # Compute task_distance_{ij}
            task_distance_ij = math.hypot(task.x - driver.x, task.y - driver.y)
            sensitivity_j = driver.sensitivity

            # Coefficients from the probability function
            gamma_ij = -(-1.451977 + 1.247782 * sensitivity_j - 0.01852485 * (delivery_distance_i / 283) - 3.893085 * (task_distance_ij / 283))
            delta_ij = 0.04330204

            # Adjusted cost from previous period (assuming c_i remains constant within the period)
            c_i_prev = c_i

            # Compute C_{ij}^* using the Lambert W formula
            try:
                exponent = gamma_ij + delta_ij * c_i_prev - 1
                W_input = np.exp(exponent)
                W_value = lambertw(W_input, k=0).real  # Use the principal branch k=0
                C_star = (-W_value - delta_ij * c_i + 1) / delta_ij
                # Ensure C_star is within feasible bounds
                C_star = max(min_compensation, min(C_star, c_i))  # Compensation cannot be less than min_compensation

                # Compute P_{ij}(C^*_{ij})
                P_ij_star = probability_function(C_star, sensitivity_j, delivery_distance_i, task_distance_ij)

                # Store the values
                expected_profit = P_ij_star * (prof_driver_cost - C_star)
                c_star[i, j] = expected_profit  # Expected profit
                P_star[i, j] = P_ij_star
                C_star_matrix[i, j] = C_star
            except Exception as e:
                # If computation fails, use fallback or set to zero
                print(f"Error computing C^* for Task {task.task_id}, Driver {driver.driver_id}: {e}")
                C_star = min_compensation
                P_ij_star = probability_function(C_star, sensitivity_j, delivery_distance_i, task_distance_ij)
                expected_profit = P_ij_star * (prof_driver_cost - C_star)
                c_star[i, j] = expected_profit
                P_star[i, j] = P_ij_star
                C_star_matrix[i, j] = C_star

    return c_star, P_star, C_star_matrix

# Function for Gurobi optimization model
def solve_optimization(tasks, drivers, c_star):
    # Create a Gurobi model
    model = Model("task_assignment_extended")
    model.setParam('OutputFlag', 0)  # Suppress Gurobi output for clarity

    # Sets
    I = range(len(tasks))    # Set of tasks
    J = range(len(drivers))  # Set of drivers

    # Decision variables
    x = model.addVars(I, J, vtype=GRB.BINARY, name="x")  # Assignment variables

    # Constraints
    # Each driver is assigned at most one task
    model.addConstrs((quicksum(x[i, j] for i in I) <= 1 for j in J), "driver_assignment")

    # Each task is assigned at most once
    model.addConstrs((quicksum(x[i, j] for j in J) <= 1 for i in I), "task_assignment")

    # Objective function: Maximize total expected profit
    obj = quicksum(c_star[i, j] * x[i, j] for i in I for j in J)
    model.setObjective(obj, GRB.MAXIMIZE)

    # Optimize the model
    model.optimize()

    # Retrieve the results
    assignments = []
    if model.status == GRB.OPTIMAL:
        for i in I:
            for j in J:
                if x[i, j].X > 0.5:
                    assignments.append((tasks[i], drivers[j], i, j))
    else:
        print("No optimal solution found.")

    return assignments

# Simulation function
def run_simulation(tasks, drivers, n_pers, t_average, model_type='bayesian'):
    # Initialize driver pool and task pool
    driver_pool = [BayesianDriver.from_dict(driver.to_dict()) for driver in drivers]
    task_pool = [Task.from_dict(task.to_dict()) for task in tasks]

    # If standard model, set all drivers' sensitivity to initial and do not update Bayesian parameters
    if model_type == 'standard':
        for d in driver_pool:
            # Keep sensitivity fixed at initial
            d.sensitivity = d.initial_sensitivity if d.initial_sensitivity is not None else 1.0

    # Initialize performance metrics
    total_tasks = 0
    completed_tasks = 0
    total_profit = 0
    total_offered_tasks = 0
    total_accepted_tasks = 0
    total_driver_utilization = 0

    # Simulate over periods
    for current_period in range(1, n_pers + 1):
        # Update driver availability based on task completion
        for driver in driver_pool:
            if not driver.available and driver.end_time <= current_period:
                driver.available = True

        # Get available drivers
        available_drivers = [driver for driver in driver_pool if driver.available]

        # Get tasks that are available in the current period
        current_tasks = [task for task in task_pool if task.start_time <= current_period <= task.end_time and task.available]

        # Remove expired tasks
        expired_tasks = [task for task in task_pool if task.end_time < current_period and task.available]
        for task in expired_tasks:
            task.available = False

        # Update total tasks
        total_tasks += len(current_tasks)

        # Compute c^* for current tasks and available drivers
        if current_tasks and available_drivers:
            c_star, P_star, C_star_matrix = compute_cij_star(current_tasks, available_drivers, current_period)

            # Solve optimization problem
            assignments = solve_optimization(current_tasks, available_drivers, c_star)

            # Process assignments
            for task, driver, i, j in assignments:
                C_star = C_star_matrix[i, j]
                P_ij = P_star[i, j]
                # Simulate driver acceptance
                if random.random() <= P_ij:
                    driver.available = False
                    driver.end_time = current_period + t_average
                    task.available = False
                    completed_tasks += 1
                    profit = prof_driver_cost - C_star
                    total_profit += profit
                    total_driver_utilization += t_average
                    total_accepted_tasks += 1
                    # Update driver parameters
                    if model_type == 'bayesian':
                        driver.update_parameters_bayesian(accepted=True)
                    else:
                        driver.update_parameters_standard()
                else:
                    if model_type == 'bayesian':
                        driver.update_parameters_bayesian(accepted=False)
                    else:
                        driver.update_parameters_standard()
                    # The task remains available for future periods until it expires

            # Update total offered tasks
            total_offered_tasks += len(assignments)
        else:
            assignments = []

    # Performance Metrics Calculation
    task_completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
    driver_utilization_rate = total_driver_utilization / (len(drivers) * n_pers) if len(drivers) > 0 else 0
    average_profit_per_task = total_profit / completed_tasks if completed_tasks > 0 else 0
    average_driver_acceptance_rate = total_accepted_tasks / total_offered_tasks if total_offered_tasks > 0 else 0
    unassigned_tasks = total_tasks - completed_tasks

    # Collect performance indicators
    performance = {
        'Total Tasks': total_tasks,
        'Completed Tasks': completed_tasks,
        'Task Completion Rate': task_completion_rate,
        'Driver Utilization Rate': driver_utilization_rate,
        'Average Profit per Task': average_profit_per_task,
        'Average Driver Acceptance Rate': average_driver_acceptance_rate,
        'Unassigned Tasks': unassigned_tasks,
        'Total Profit': total_profit,
        'Model Type': model_type
    }

    return performance

# Main function to run simulations
def main():
    # Simulation parameters
    t_arrs = [5, 10, 15]
    d_arrs = t_arrs
    t_avgs = [1, 2, 3]
    d_avgs = t_avgs
    n_pers = 100
    seed = 1000
    n_instances = 100  # Adjust as needed

    # For simplicity, let's run one configuration
    t_arrival = t_arrs[0]
    d_arrival = d_arrs[0]
    t_average = t_avgs[0]
    d_average = d_avgs[0]

    # Sensitivity values
    initial_sensitivity_values = np.linspace(0.5, 2.0, 5)

    # Step 1: Generate and save instances (only once)
    print("Generating and saving instances...")
    for instance_number in range(1, n_instances + 1):
        tasks, drivers = generate_instances(n_pers, t_arrival, d_arrival, t_average, d_average, seed + instance_number)
        save_instances_to_csv(tasks, drivers, instance_number)

    # Run simulations for both models
    models = ['standard', 'bayesian']

    for model_type in models:
        print(f"Running simulations for model: {model_type}...")
        all_results = []
        for initial_sensitivity in initial_sensitivity_values:
            print(f"Running simulations for initial sensitivity value: {initial_sensitivity}")
            sensitivity_results = []
            for instance_number in range(1, n_instances + 1):
                # Read instances and set initial sensitivity
                tasks, drivers = read_instances_from_csv(instance_number, initial_sensitivity)

                # Run simulation
                performance = run_simulation(tasks, drivers, n_pers, t_average, model_type=model_type)
                performance['Initial Sensitivity'] = initial_sensitivity
                sensitivity_results.append(performance)

            # Collect results
            df = pd.DataFrame(sensitivity_results)
            all_results.append(df)

        # Combine results from all sensitivity values
        combined_results = pd.concat(all_results, ignore_index=True)

        # Step 3: Save combined results
        combined_results.to_csv(f'{model_type}_sensitivity_analysis_results.csv', index=False)

        # Step 4: Create box plots for visualization
        print(f"Creating box plots for model: {model_type}...")
        metrics = ['Task Completion Rate', 'Driver Utilization Rate', 'Average Profit per Task', 'Average Driver Acceptance Rate', 'Total Profit']

        # Set the style for seaborn
        sns.set(style="whitegrid")

        # Create box plots for each metric
        for metric in metrics:
            plt.figure(figsize=(10, 7))
            ax = sns.boxplot(x='Initial Sensitivity', y=metric, data=combined_results, palette="Set2")
            ax.set_title(f'Impact of Initial Sensitivity on {metric} ({model_type.capitalize()} Model)')
            ax.set_xlabel('Initial Sensitivity')
            plt.tight_layout()
            # Save the figure
            plt.savefig(f'{metric.replace(" ", "_")}_{model_type}_sensitivity_boxplot.png')
            plt.close()
            print(f"Box plot for {metric} ({model_type} model) saved as {metric.replace(' ', '_')}_{model_type}_sensitivity_boxplot.png")

if __name__ == "__main__":
    main()
