using ProductApi.Services;  // Asegúrate de que este espacio de nombre esté presente


var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
// Registramos el servicio ProductService como un servicio HTTP
builder.Services.AddHttpClient<IProductService, ProductService>();

// Configuramos los controladores
builder.Services.AddControllers();

// Añadimos la generación de Swagger
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

app.UseAuthorization();

app.MapControllers();

// Configurar para que escuche solo en IPv4
app.Urls.Add("http://0.0.0.0:8080");

app.Run();
