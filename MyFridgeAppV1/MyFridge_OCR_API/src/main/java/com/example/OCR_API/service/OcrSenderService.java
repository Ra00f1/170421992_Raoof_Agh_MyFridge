package com.example.OCR_API.service;

import com.example.OCR_API.model.OcrPayload;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.FileTime;
import java.util.stream.Stream;
import java.util.Comparator;

@Service
public class OcrSenderService 
{

    private static final String UPLOADS_DIR = "uploads"; // klasör adı
    private static final String FILE_PREFIX = "ocr_";
    private static final String FILE_EXTENSION = ".json";

    public void sendLatestParsedDataToMainApi() 
    {
        try {
            String latestFilePath = findLatestJsonFilePath();
            if (latestFilePath == null) 
            {
                System.err.println("Hiçbir JSON dosyası bulunamadı.");
                return;
            }

            ObjectMapper mapper = new ObjectMapper();

            // dosya doğrudan OcrPayload formatında okunur
            OcrPayload payload = mapper.readValue(
                Paths.get(latestFilePath).toFile(),
                OcrPayload.class
            );

            // HTTP isteği hazırlanır
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<OcrPayload> request = new HttpEntity<>(payload, headers);

            String mainApiUrl = "http://localhost:8080/api/products/upload";
            RestTemplate restTemplate = new RestTemplate();
            ResponseEntity<String> response = restTemplate.postForEntity(mainApiUrl, request, String.class);

            System.out.println("Yanıt: " + response.getBody());

        } 
        catch (IOException e) 
        {
            e.printStackTrace();
        }
    }

    private String findLatestJsonFilePath() throws IOException 
    {
        try (Stream<Path> files = Files.list(Paths.get(UPLOADS_DIR))) 
        {
            return files
                    .filter(p -> p.getFileName().toString().startsWith(FILE_PREFIX))
                    .filter(p -> p.getFileName().toString().endsWith(FILE_EXTENSION))
                    .max(Comparator.comparing(p -> 
                    {
                        try 
                        {
                            return Files.getLastModifiedTime(p);
                        } 
                        catch (IOException e) 
                        {
                            return FileTime.fromMillis(0);
                        }
                    }))
                    .map(Path::toString)
                    .orElse(null);
        }
    }
}
