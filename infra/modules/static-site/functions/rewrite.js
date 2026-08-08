/**
 * Resolve directory URLs to their index document.
 *
 * A private bucket behind Origin Access Control is reached through the S3 REST
 * endpoint, and the REST endpoint has no notion of an index document - that is
 * a feature of the (public, HTTP-only) website endpoint we deliberately do not
 * use. Without this, "/" asks S3 for the empty key and "/panel-a/" asks for a
 * prefix, and both come back as errors.
 *
 * A CloudFront Function rather than Lambda@Edge: this is string manipulation
 * measured in microseconds, it runs at every edge location, and it costs about
 * a sixth as much.
 */
function handler(event) {
  var request = event.request;
  var uri = request.uri;

  if (uri.endsWith('/')) {
    request.uri = uri + 'index.html';
  } else if (!uri.includes('.')) {
    // Extensionless path: Next's export wrote it as a directory.
    request.uri = uri + '/index.html';
  }

  return request;
}
