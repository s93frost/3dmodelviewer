class MTLParser:
        def __init__(self, mtl_path):
            self.mtl_path = mtl_path
            self.materials = {}


        def parse(self):
            current_material = None

            with open(self.mtl_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    # Remove inline comments
                    if '#' in line:
                        line = line.split('#')[0].strip()

                    tokens = line.split()
                    if not tokens:
                        continue

                    if tokens[0] == 'newmtl':
                        current_material = tokens[1]
                        self.materials[current_material] = {}
                    elif current_material:
                        key = tokens[0]
                        value = tokens[1:]
                        self.materials[current_material][key] = value

            return self.materials