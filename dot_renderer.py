from graphviz import Source

dot_code = '''
digraph Software_Architecture {
  rankdir=LR;

  Customer;
  React_Frontend;
  Spring_Boot_Backend;
  MySQL_Database;
  Redis;
  External_Payment_API;

  Customer -> React_Frontend;
  React_Frontend -> Spring_Boot_Backend;
  Spring_Boot_Backend -> MySQL_Database;
  Spring_Boot_Backend -> Redis;
  Spring_Boot_Backend -> External_Payment_API;
}
'''
output_path = "outputs/architecture"

graph = Source(dot_code)
graph.render(output_path, format="png", cleanup=True)

print("Diagram generated successfully!")
print(f"Saved as: {output_path}.png")