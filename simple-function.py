import math

def analyze_numbers(numbers):
    # Remove duplicates and sort
    unique_sorted = sorted(set(numbers))

    # Basic stats
    total = sum(unique_sorted)
    avg = total / len(unique_sorted)
    smallest = min(unique_sorted)
    largest = max(unique_sorted)

    # Factorial of largest number
    fact = math.factorial(largest)

    # Return results in a dictionary
    return {
        "original": numbers,
        "unique_sorted": unique_sorted,
        "sum": total,
        "average": avg,
        "min": smallest,
        "max": largest,
        "factorial_of_max": fact
    }

# Example usage
nums = [5, 3, 2, 5, 7, 3, 9, 7]
result = analyze_numbers(nums)

# Print results nicely
for key, value in result.items():
    print(f"{key}: {value}")
