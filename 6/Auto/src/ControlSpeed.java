import net.sourceforge.jFuzzyLogic.FIS;

public class ControlSpeed {
    public static void main(String[] args) throws Exception {
        // Загружаем 'FCL' файл.
        String fileName = "speed.fcl";
        FIS fis = FIS.load(fileName, true);

        // Ошибка при загрузке ?
        if( fis == null )
        {
            System.err.println("Ошибка при загрузке файла: '" + fileName + "'");
            return;
        }

        // Показываем.
        fis.chart();

        // Задаем значения входных переменных.
        fis.setVariable("speed", 10);
        fis.setVariable("temperatureOnStreet", 18);
        fis.setVariable("requestedSpeed", 100);
        fis.setVariable("masaAuto", 2);

        // Вычисляем.
        fis.evaluate();

        // Печатаем информацию о выходной перменной.
        System.out.println(fis.getVariable("toSpeed").toString());
        System.out.println(fis.getVariable("toSpeedUp").toString());

        // Печатаем вещественное значение последней дефаззификации выходной переменной.
        System.out.println(fis.getVariable("toSpeed").getValue());
        System.out.println(fis.getVariable("toSpeedUp").getValue());

        // Показываем график выходной переменной.
        fis.getVariable("toSpeed").chartDefuzzifier(true);
        fis.getVariable("toSpeedUp").chartDefuzzifier(true);

        // Печатаем набор правил.
        System.out.println(fis);
    }
}
