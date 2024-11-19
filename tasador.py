from core import CBR, CBR_DEBUG

import cbrkit



class TasadorCBR(CBR):
    def __init__(self, base_de_casos, 
                       num_casos_similares, 
                       taxonomia_colors="datos/paint_color.yaml", 
                       taxonomia_manufacturer="datos/manufacturer.yaml",
                       umbral_precio=10,
                       debug = False):
        super().__init__(base_de_casos, num_casos_similares)
        #CBR.__init__(self, base_de_casos, num_casos_similares)
        if debug:
            self.DEBUG = CBR_DEBUG(self.prettyprint_caso)
        else: 
            self.DEBUG = None
        self.umbral_precio = umbral_precio
        self.retriever = self.inicializar_retriever(num_casos_similares, taxonomia_colors, taxonomia_manufacturer)    
        
    def inicializar_retriever(self, num_casos_similares, 
                              taxonomia_colors, 
                              taxonomia_manufacturer):
        
        # ejemplos de similaridades sobre taxonomias (colores y marcas)
        color_similarity = cbrkit.sim.strings.taxonomy.load(taxonomia_colors, 
                                                    cbrkit.sim.strings.taxonomy.wu_palmer())

        manufacturer_similarity = cbrkit.sim.strings.taxonomy.load(taxonomia_manufacturer, 
                                                    cbrkit.sim.strings.taxonomy.wu_palmer())

        # ejemplo de similaridad para "objeto anidado" model (marca + modelo)
        model_similarity = cbrkit.sim.attribute_value(
                attributes={
                    "manufacturer": manufacturer_similarity,
                    "make": cbrkit.sim.strings.levenshtein()
                },
                aggregator=cbrkit.sim.aggregator(pooling="mean")
            )

        # similaridad de "millas" (ejemplo de funcion de similaridad propia)
        def miles_similarity(x,y):
            diff = abs(x-y)
            if diff < 10000:
                return 1.0
            elif diff < 25000:
                return 0.8
            elif diff < 50000:
                return 0.5
            elif diff < 100000:
                return 0.2
            else:
                return 0.0 

        # ejemplo de similaridad para "objeto anidado" engine (traccion + combustible + transmision) con un agragador con ponderacion
        engine_similarity = cbrkit.sim.attribute_value(
                attributes={
                    "drive": cbrkit.sim.generic.table([("4w", "fw", 0.6), 
                                                       ("4w", "rw", 0.4),
                                                       ("fw", "rw", 0.2)], 
                                                        symmetric=True, default=1.0), # Solo 3 valores => concidencia -> caso default
                    "fuel": cbrkit.sim.strings.levenshtein(),
                    "transmission": cbrkit.sim.generic.equality(),
                },
                aggregator=cbrkit.sim.aggregator(pooling="mean",
                                                 pooling_weights={"drive": 0.3,
                                                                  "fuel":0.6,
                                                                  "transmission": 0.1})
            )

        # funcion de similaridad completa
        case_similarity = cbrkit.sim.attribute_value(
                            attributes={
                                "type": cbrkit.sim.generic.equality(),
                                "title_status": cbrkit.sim.generic.equality(),
                                "model" : model_similarity,
                                "engine" : engine_similarity, 
                                "paint_color": color_similarity,
                                "year": cbrkit.sim.numbers.linear(max=20),
                                "miles": miles_similarity
                            },
                            aggregator=cbrkit.sim.aggregator(pooling="mean"),
                        )
        # creacion del retriever
        retriever = cbrkit.retrieval.build(case_similarity, limit=num_casos_similares)
        return retriever
        
    def prettyprint_caso(self, caso, meta = None):
        prettyprint_caso = "{} {}, año: {}, millas: {}, precio: {}".format(caso['model']['manufacturer'], 
                                              caso['model']['make'], caso['year'], caso['miles'], caso['price'])
        if (meta is None) and '_meta' in caso:
            meta = caso['_meta']
        
        if meta is not None:    
            pretty_print_meta = "[META: id: {}, precio_real: {}, precio_predicho: {}, exito: {}, corregido: {}]".format(meta['id'], 
                                                                                    meta['price_real'], meta['price_predicho'],
                                                                                    meta['exito'],meta['corregido'])
            prettyprint_caso = prettyprint_caso + " -> " + pretty_print_meta        
    
        return prettyprint_caso
    
        
    def inicializar_caso(self, caso, id = None):
        # inicializar atributo _meta, anotando id si lo hay         
        super().inicializar_caso(caso, id)
        
        # inicializar metadatos del caso para el problema de tasacion
        if 'price' in caso:
             # si el caso ya tiene asignado un 'price', se anota el precio real en los metadatos del caso     
            caso['_meta']['price_real'] = caso['price']
        else:
            caso['_meta']['price_real'] = 0.0
        caso['_meta']['price_predicho'] = 0.0
        caso['_meta']['exito'] = False
        caso['_meta']['corregido'] = False

        return caso

        
    def recuperar(self, caso_a_resolver):    
        result = cbrkit.retrieval.apply(self.base_de_casos, caso_a_resolver, self.retriever)
        casos_similares = []
        similaridades = []
        for i in result.ranking:
            casos_similares.append(self.base_de_casos[i])
            similaridades.append(result.similarities[i].value)
        
        # DEBUG
        if self.DEBUG : self.DEBUG.debug_recuperar(caso_a_resolver, casos_similares, similaridades)
          
        return (casos_similares, similaridades)
    
    def reutilizar(self, caso_a_resolver, casos_similares, similaridades):
        # calcular precio como media de los precios de los casos similares
        price_acc = 0.0
        for c in casos_similares:
            price_acc = price_acc + c['price']
        price_predicho = price_acc / len(casos_similares)
        
        # copiar el caso original
        caso_resuelto = dict(caso_a_resolver)
        
        # ajustar metadatos (almacenar precio real y precio predicho)
        if 'price' in caso_resuelto:
            caso_resuelto['_meta']['price_real'] = caso_resuelto['price']
        else:
            caso_resuelto['_meta']['price_real'] = 0.0
        caso_resuelto['_meta']['price_predicho'] = price_predicho   

        # actualizar precio
        caso_resuelto['price'] = price_predicho
        
        # DEBUG
        if self.DEBUG : self.DEBUG.debug_reutilizar(caso_resuelto)
        
        return caso_resuelto
                 
        
    def revisar(self, caso_resuelto, caso_a_resolver=None, casos_similares=None, similaridades=None):
        # simula revisión por experto (comparando precio predicho con precio real del caso)
        # marca el caso resuelto como éxito si la diferencia de precio predicho con el real es inferior al 5%
        precio_real = caso_resuelto['_meta']['price_real']
        precio_predicho = caso_resuelto['_meta']['price_predicho']
        diff = abs(precio_real-precio_predicho)
        diff_100 = 100.0 * (diff/precio_real) # % de error sobre precio real
        
        # copiar el caso recibido
        caso_revisado = dict(caso_resuelto)

        if diff_100 <= self.umbral_precio:
            # caso de exito => marcar exito en metadatos y mantener precio predicho
            caso_revisado['_meta']['exito'] = True
            caso_revisado['_meta']['corregido'] = False
            caso_revisado['price'] = caso_revisado['_meta']['price_predicho']
        else:
            # caso de fracaso => marcar fracaso en metadatos y corregir con precio real
            caso_revisado['_meta']['exito'] = False   
            caso_revisado['_meta']['corregido'] = True
            caso_revisado['price'] = caso_revisado['_meta']['price_real']

        # DEBUG    
        if self.DEBUG : self.DEBUG.debug_revisar(caso_revisado, 
                                               es_exito=caso_revisado['_meta']['exito'], 
                                               es_corregido=caso_revisado['_meta']['corregido'])
        
        return caso_revisado

    def retener(self, caso_revisado, caso_a_resolver=None, casos_similares=None, similaridades=None):
        es_retenido = False
        
        # retener solo los casos que se hayan corregido 
        if caso_revisado['_meta']['corregido']:
            self.base_de_casos[caso_revisado['_meta']['id']] = caso_revisado
            es_retenido = True
        
        # DEBUG    
        if self.DEBUG : self.DEBUG.debug_retener(caso_revisado, es_retenido=es_retenido)