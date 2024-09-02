using System.Net.Http;
using System.Text;
using System.Text.Json;

namespace ProductApi.Services
{
    public interface IProductService
    {
        Task<string> GetRecommendations(int productId);
    }

    public class ProductService : IProductService
    {
        private readonly HttpClient _httpClient;
        private readonly string _baseUrl = "http://python_service:5000/recommend/category/";

        public ProductService(HttpClient httpClient)
        {
            _httpClient = httpClient;
        }

        public async Task<string> GetRecommendations(int productId)
        {
            // Construir la URL completa
            var recommendationServiceUrl = $"{_baseUrl}{productId}";
            
            // Realizar la solicitud GET a la API externa
            var response = await _httpClient.GetAsync(recommendationServiceUrl);

            // Verificar si la respuesta fue exitosa
            response.EnsureSuccessStatusCode();

            // Devolver el contenido de la respuesta como cadena
            return await response.Content.ReadAsStringAsync();
        }
    }
}
