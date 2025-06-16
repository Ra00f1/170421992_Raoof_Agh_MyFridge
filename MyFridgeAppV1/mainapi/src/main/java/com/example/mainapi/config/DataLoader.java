package com.example.mainapi.config;

import com.example.mainapi.model.CategoryKeyword;
import com.example.mainapi.repository.CategoryKeywordRepository;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class DataLoader 
{

    @Bean
    public CommandLineRunner loadKeywords(CategoryKeywordRepository repository) 
    {
        return args -> 
        {
            if (repository.count() == 0) 
            {
                repository.save(new CategoryKeyword("yoğurt", "süt ürünleri"));
                repository.save(new CategoryKeyword("cola", "içecek"));
                repository.save(new CategoryKeyword("piliç döner", "et ürünleri"));
                repository.save(new CategoryKeyword("karam", "tatlı"));
                repository.save(new CategoryKeyword("pirinç", "bakliyat"));
                repository.save(new CategoryKeyword("ekmek", "fırın ürünleri"));
            }
        };
    }
}

