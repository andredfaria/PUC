package projetoweka;

public class ProjetoWeka {

    public static void main(String[] args) throws Exception {
// salve o arquivo íris.arff na mesma pasta do projeto 
        path = "iris.arff";
        weka wekaInstance = new weka(path);
        wekaInstance.leDados();
        wekaInstance.imprimeDados();
        wekaInstance.arvoreDeDecisaoJ48();
        wekaInstance.InstanceBased();

    }

    static String path;
}
