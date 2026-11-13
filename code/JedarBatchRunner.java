import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;

public class JedarBatchRunner {
    public static void main(String[] args) throws Exception {
        String[] suffixes = Jedar.readsimple("suffixes.txt");
        if (suffixes.length == 0) {
            throw new IllegalStateException("Jedar suffixes.txt is missing or empty");
        }
        String bom = String.valueOf((char) 65279);
        if (suffixes[0].startsWith(bom)) {
            suffixes[0] = suffixes[0].substring(1);
        }

        try (BufferedReader input = new BufferedReader(
                     new InputStreamReader(System.in, StandardCharsets.UTF_8));
             PrintWriter output = new PrintWriter(
                     new OutputStreamWriter(System.out, StandardCharsets.UTF_8), true)) {
            String word;
            while ((word = input.readLine()) != null) {
                HashMap<String, Integer> candidates = Jedar.suffixRemover(word, suffixes);
                String stem = Jedar.selectStem(word, candidates, 3);
                boolean empty = stem.isEmpty();
                output.println((empty ? "1" : "0") + "\t" + (empty ? word : stem));
            }
        }
    }
}
