class CBR:
    def __init__(self, base_de_casos, num_casos_similares = 1 ):
        self.base_de_casos = base_de_casos
        self.num_casos_similares= num_casos_similares
    
    def inicializar_caso(self, caso, id = None):
        # crear un atributo _meta para vincular metadatos durante el ciclo CBR
        caso['_meta'] = {}
        
        if id is not None:
            caso['_meta']['id'] = id
        else:
            # comprobar si el caso tiene un identificador interno, tomar nota del mismo
            if 'id' in caso:
                caso['_meta']['id'] = caso['id']
            if 'uuid' in caso:
                caso['_meta']['id'] = caso['uuid']
    
        return caso
    
    def recuperar(self, caso_a_resolver):
        pass
    
    def reutilizar(self,  caso_a_resolver, casos_similares, similaridades):
        pass

    def revisar(self, caso_resuelto, caso_a_resolver=None, casos_similares=None, similaridades=None):
        pass

    def retener(self, caso_revisado, caso_a_resolver=None, casos_similares=None, similaridades=None):
        pass
    
    def ciclo_cbr(self, caso_a_resolver, id_caso=None):
        caso_a_resolver = self.inicializar_caso(caso_a_resolver, id=id_caso)
        (casos_similares, similaridades) = self.recuperar(caso_a_resolver)#, base_de_casos, similaridad)
        caso_resuelto = self.reutilizar(caso_a_resolver, casos_similares, similaridades)
        caso_revisado = self.revisar(caso_resuelto, caso_a_resolver, casos_similares, similaridades)    
        self.retener(caso_revisado, caso_a_resolver, casos_similares, similaridades)
        return caso_revisado
    

class CBR_DEBUG:
    def __init__(self, prettyprint_caso):
        self.prettyprint_caso = prettyprint_caso

    def debug_mensaje(self, etiqueta, mensaje=""):
        print("[{}] {}".format(etiqueta, mensaje))

    def debug_recuperar(self, caso, similares, scores):    
        self.debug_mensaje("DEBUG.recuperar",mensaje="CASO A RESOLVER: "+ self.prettyprint_caso(caso))
        self.debug_mensaje("DEBUG.recuperar",mensaje="CASOS SIMILARES (total: {}, similaridad max.:{})".format(len(similares), scores[0]))

        count = 1
        for (c, score) in zip (similares, scores):
            self.debug_mensaje("DEBUG.recuperar",mensaje="- [{}] {}  (Similaridad: {})".format(count, self.prettyprint_caso(c), score))
            count = count + 1
        self.debug_mensaje("DEBUG.recuperar")
              
    def debug_reutilizar(self, caso_resuelto):    
        self.debug_mensaje("DEBUG.reutilizar", mensaje="CASO RESUELTO: "+ self.prettyprint_caso(caso_resuelto))
        self.debug_mensaje("DEBUG.retutilzar")

    def debug_revisar(self, caso_revisado, es_exito=None, es_corregido=None):    
        self.debug_mensaje("DEBUG.revisar", mensaje="CASO REVISADO: " + self.prettyprint_caso(caso_revisado))
        if es_exito is not None:
            self.debug_mensaje("DEBUG.revisar", mensaje="- exito: {}".format(es_exito))
        if es_corregido is not None:
            self.debug_mensaje("DEBUG.revisar", mensaje="- corregido: {}".format(es_corregido))
        self.debug_mensaje("DEBUG.revisar")

    def debug_retener(self, caso_retenido, es_retenido=None):    
        self.debug_mensaje("DEBUG.retener", mensaje="CASO RETENIDO: "+ self.prettyprint_caso(caso_retenido))
        if es_retenido is not None:
            if es_retenido and ('id' in caso_retenido):
                self.debug_mensaje("DEBUG.retener","- retenido: {} con id manual: {}".format(es_retenido, caso_retenido['id']))
            else:
                self.debug_mensaje("DEBUG.retener","- retenido: {}".format(es_retenido))
        self.debug_mensaje("DEBUG_retener")