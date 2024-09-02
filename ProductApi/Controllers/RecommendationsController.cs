using Microsoft.AspNetCore.Mvc;
using ProductApi.Services;

namespace ProductApi.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class RecommendationsController : ControllerBase
    {
        private readonly IProductService _productService;

        public RecommendationsController(IProductService productService)
        {
            _productService = productService;
        }

        [HttpGet("{productId}")]
        public async Task<IActionResult> GetRecommendations(int productId)
        {
            // Llamar al servicio para obtener las recomendaciones
            var recommendations = await _productService.GetRecommendations(productId);
            return Ok(recommendations);
        }
    }
}
