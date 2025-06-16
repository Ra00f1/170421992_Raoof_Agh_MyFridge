package com.example.mainapi.service;

import com.example.mainapi.model.Recipe;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
public class RecipeService 
{

    private final DbApiSenderService dbApiSenderService;

    public RecipeService(DbApiSenderService dbApiSenderService) 
    {
        this.dbApiSenderService = dbApiSenderService;
    }

    public void sendToDbApi(Recipe recipe, Long userId) 
    {
        String url = "http://localhost:8082/api/recipe/add/" + userId;
        org.springframework.web.client.RestTemplate restTemplate = new org.springframework.web.client.RestTemplate();
        restTemplate.postForObject(url, recipe, String.class);

        // Tarif başarıyla gönderildiyse eksiltme işlemi yapılır
        eksiltMalzemeleri(recipe.getMalzemeler(), userId);
    }

    public void eksiltMalzemeleri(List<String> malzemeler, Long userId) 
    {
        for (String malzeme : malzemeler) 
        {
            try 
            {
                // Örnek: "su - 2 L"
                String[] parts = malzeme.split(" - ");
                if (parts.length < 2) continue;

                String productName = parts[0].trim().toLowerCase();
                String[] miktarParts = parts[1].trim().split(" ");
                if (miktarParts.length < 2) continue;

                int miktar = Integer.parseInt(miktarParts[0].trim());

                dbApiSenderService.eksiltUserFridge(userId, productName, miktar);

            } 
            catch (Exception e) 
            {
                System.err.println("Eksiltme hatası: " + malzeme + " - " + e.getMessage());
            }
        }
    }

}
