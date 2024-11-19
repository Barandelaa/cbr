import json
import random
import argparse


def main(args):
    # Cargar el array JSON desde un archivo
    path_base_casos = args.base_casos 
    with open(path_base_casos, 'r') as file:
        casos = json.load(file)

    # Barajar el array
    random.shuffle(casos)

    casos_a_resolver = casos[:args.num_elementos]  # Primeros elementos
    nueva_base_casos = casos[args.num_elementos:]  # Elementos restantes

    # Crear nombre de los ficheros de salida
    if args.destino:
        if args.destino.endswith('.json'):
            nombre_base = args.destino
        else:
            nombre_base = args.destino+".json"
    else:
        nombre_base = path_base_casos
        
    # Guardar JSON
    with open(nombre_base.replace('.json', '.base_casos.json'), 'w') as file:
        json.dump(nueva_base_casos, file, indent=2)

    with open(nombre_base.replace('.json', '.casos_a_resolver.json'), 'w') as file:
        json.dump(casos_a_resolver, file, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extrae n elementos al azar de una base de casos JSON')
    parser.add_argument('-b', '--base_casos',
                        type=str, required=True,
                        help='Path al fichero JSON con la base de casos (lista de objetos)' )
    parser.add_argument('-n', '--num_elementos',
                        type=int, default=10, required=False,
                        help='Numero de elementos a extraer del array JSON',
                        )
    parser.add_argument('-d', '--destino',
                        type=str, required=False,
                        help='Nombre/ruta con el que crear los dos archivos resultantes (si no se indica usa el nombre del fichero JSON de entrada)',
                        )
    args = parser.parse_args()
    main(args)