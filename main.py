
import cbrkit
import argparse
import pprint
import random

from tasador import TasadorCBR



def main(args):
    # leer datos
    base_de_casos = cargar_base_casos(args.base_casos)
    casos_a_resolver = cargar_base_casos(args.casos)

    # incializar tasador
    tasador = TasadorCBR(base_de_casos, num_casos_similares=args.num_similares, 
                                        debug=args.debug)

    # procesar casos a resolver
    contador_exitos = 0
    for id_caso in casos_a_resolver.keys():
        caso = casos_a_resolver[id_caso]
        # establecer un nuevo id del caso manualmente
        nuevo_id = 100000+id_caso
        caso_resuelto = tasador.ciclo_cbr(caso, id_caso=nuevo_id)
        if caso_resuelto['_meta']['exito']:
            contador_exitos = contador_exitos + 1
        print("--- CASO RESUELTO ---")
        pprint.pprint(caso_resuelto, width=2)
        print("---------------------")

    print("RESUMEN: {} exito/s de {} casos procesados".format(contador_exitos, len(casos_a_resolver)))

    
    
    
def cargar_base_casos(path):
    return cbrkit.loaders.json(path)
 
 
def extraer_casos_a_resolver(base_de_casos, cantidad):
    casos_a_resolver = []
    indices_aleatorios = random.sample(list(base_de_casos.keys()), cantidad)
    
    for i in indices_aleatorios:
        caso = base_de_casos.pop(i)
        casos_a_resolver.append(caso)
    return casos_a_resolver    
        
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ejemplo de uso del modelo CBR para tasaciones de coches')
    parser.add_argument('-b', '--base_casos',
                        type=str, default='datos/cve.base_de_casos.json',  required=False,
                        help='Path al fichero JSON con la base de casos (lista de objetos coche)' )
    parser.add_argument('-r', '--casos',
                        type=str, default='datos/cve.casos_a_resolver.json', required=False,
                        help='Path al fichero JSON con los casos a resolver (lista de objetos coche)',
                        )
    parser.add_argument('-c', '--cwe',
                        type=str, default='datos/jerarquia_cwe_1000.yaml', required=False,
                        help='Path al fichero YAML con la taxonomia de colores',
                        )
    parser.add_argument('-n', '--num_similares',
                        type=int, default=5, required=False,  
                        help='Num. de casos similares a utilizar')   
    parser.add_argument('-u', '--umbral_precio',
                        type=int, default=10, required=False,  
                        help='porcentaje de error en el precio predicho admitido como correcto en la fase Revision')   

    parser.add_argument('-d','--debug',
                        action='store_true', required=False, default=False,
                        help='Activar DEBUG de fases del ciclo CBR')

    args = parser.parse_args()
    main(args)
