package org.example;

import put.dea.robustness.CCRExtremeRanks;
import put.dea.robustness.ProblemData;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class CsvExtremeRanksExporter {

    public static void main(String[] args) throws IOException {
        String inputCsvPath = args.length > 0 ? args[0] : "../input/airports.csv";
        String outputCsvPath = args.length > 1 ? args[1] : "../output/extreme_ranks.csv";

        new CsvExtremeRanksExporter().run(inputCsvPath, outputCsvPath);

        System.out.println("Done. Extreme ranks saved to: " + outputCsvPath);
    }

    public void run(String inputCsvPath, String outputCsvPath) throws IOException {
        List<DmuRow> rows = readRowsFromCsv(inputCsvPath);
        if (rows.isEmpty()) {
            throw new IllegalArgumentException("CSV contains no data rows: " + inputCsvPath);
        }

        List<String> inputNames = detectColumns(rows.get(0).rawValues().keySet(), "i");
        List<String> outputNames = detectColumns(rows.get(0).rawValues().keySet(), "o");

        ProblemDataBundle bundle = buildProblemData(rows, inputNames, outputNames);
        CCRExtremeRanks extremeRanks = new CCRExtremeRanks();

        Path outputPath = Path.of(outputCsvPath);
        if (outputPath.getParent() != null) {
            Files.createDirectories(outputPath.getParent());
        }

        try (BufferedWriter writer = Files.newBufferedWriter(outputPath)) {
            writer.write("name,best_rank,worst_rank");
            writer.newLine();

            for (int i = 0; i < rows.size(); i++) {
                int bestRank = extremeRanks.minRank(bundle.problemData(), i);
                int worstRank = extremeRanks.maxRank(bundle.problemData(), i);

                writer.write(String.join(",",
                        escapeCsv(rows.get(i).name()),
                        Integer.toString(bestRank),
                        Integer.toString(worstRank)
                ));
                writer.newLine();
            }
        }
    }

    private ProblemDataBundle buildProblemData(
            List<DmuRow> rows,
            List<String> inputNames,
            List<String> outputNames
    ) {
        double[][] inputs = new double[rows.size()][inputNames.size()];
        double[][] outputs = new double[rows.size()][outputNames.size()];
        List<String> dmuNames = new ArrayList<>();

        for (int r = 0; r < rows.size(); r++) {
            DmuRow row = rows.get(r);
            dmuNames.add(row.name());

            for (int c = 0; c < inputNames.size(); c++) {
                inputs[r][c] = parseDouble(row.rawValues().get(inputNames.get(c)), row.name(), inputNames.get(c));
            }

            for (int c = 0; c < outputNames.size(); c++) {
                outputs[r][c] = parseDouble(row.rawValues().get(outputNames.get(c)), row.name(), outputNames.get(c));
            }
        }

        ProblemData data = new ProblemData(inputs, outputs, inputNames, outputNames);
        return new ProblemDataBundle(data, dmuNames);
    }

    private List<DmuRow> readRowsFromCsv(String csvPath) throws IOException {
        List<DmuRow> rows = new ArrayList<>();

        try (BufferedReader reader = Files.newBufferedReader(Path.of(csvPath))) {
            String headerLine = reader.readLine();
            if (headerLine == null || headerLine.isBlank()) {
                throw new IllegalArgumentException("CSV is empty: " + csvPath);
            }

            String[] headers = splitCsvLine(headerLine);

            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }

                String[] parts = splitCsvLine(line);
                if (parts.length != headers.length) {
                    throw new IllegalArgumentException(
                            "Invalid CSV row. Expected " + headers.length + " columns but got " + parts.length +
                                    ". Row: " + line
                    );
                }

                Map<String, String> rawValues = new LinkedHashMap<>();
                for (int i = 0; i < headers.length; i++) {
                    rawValues.put(headers[i].trim(), unquote(parts[i].trim()));
                }

                String name = rawValues.get("name");
                if (name == null || name.isBlank()) {
                    throw new IllegalArgumentException("Encountered row with empty 'name'.");
                }

                rows.add(new DmuRow(name, rawValues));
            }
        }

        return rows;
    }

    private List<String> detectColumns(Set<String> allColumns, String prefix) {
        return allColumns.stream()
                .filter(col -> col.matches("^" + prefix + "\\d+$"))
                .sorted(Comparator.comparingInt(col -> Integer.parseInt(col.substring(1))))
                .collect(Collectors.toList());
    }

    private double parseDouble(String value, String dmuName, String colName) {
        try {
            return Double.parseDouble(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "Invalid numeric value for DMU='" + dmuName + "', column='" + colName + "': " + value, e
            );
        }
    }

    private String escapeCsv(String value) {
        if (value == null) {
            return "";
        }
        if (value.contains(",") || value.contains("\"") || value.contains("\n")) {
            return "\"" + value.replace("\"", "\"\"") + "\"";
        }
        return value;
    }

    private String unquote(String value) {
        String v = value;
        if (v.startsWith("\"") && v.endsWith("\"") && v.length() >= 2) {
            v = v.substring(1, v.length() - 1).replace("\"\"", "\"");
        }
        return v;
    }

    private String[] splitCsvLine(String line) {
        List<String> tokens = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean inQuotes = false;

        for (int i = 0; i < line.length(); i++) {
            char ch = line.charAt(i);

            if (ch == '"') {
                if (inQuotes && i + 1 < line.length() && line.charAt(i + 1) == '"') {
                    current.append('"');
                    i++;
                } else {
                    inQuotes = !inQuotes;
                }
            } else if (ch == ',' && !inQuotes) {
                tokens.add(current.toString());
                current.setLength(0);
            } else {
                current.append(ch);
            }
        }

        tokens.add(current.toString());
        return tokens.toArray(new String[0]);
    }

    private record DmuRow(String name, Map<String, String> rawValues) {}
    private record ProblemDataBundle(ProblemData problemData, List<String> dmuNames) {}
}
